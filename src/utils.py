import base64
import json
import os
from PIL import Image
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
from skimage import draw
from tensorflow.keras import backend as K


def resize_and_crop(img, min_side, centering=(0.5, 0.5), rgb=True):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    w, h = img.size
    if h < w:
        img = img.resize((int(w * (min_side / h)) + 1, min_side))
        img = ImageOps.fit(img, (min_side, min_side), centering=centering)
    else:
        img = img.resize((min_side, int(h * (min_side / w)) + 1))
        img = ImageOps.fit(img, (min_side, min_side), centering=centering)
    
    if rgb:
        img = img.convert("RGB")
        return np.array(img)
    else:
        # This is binary, because type 1 is broken in pillow
        img = img.convert("L")
        arr = np.array(img)
        arr[arr > 0] = 1
        return arr

def parse_labelme_file(sample_path):
    with open(sample_path, 'r') as f:
        sample = json.load(f)
        data = base64.b64decode(sample["imageData"])
        arr = np.array(Image.open(BytesIO(data)))
        masks = []
        for shape in sample["shapes"]:
            col_coords, row_coords = zip(*shape["points"])
            fill_row_coords, fill_col_coords = draw.polygon(
                row_coords,
                col_coords,
                (sample["imageHeight"], sample["imageWidth"])
            )
            mask = np.zeros((sample["imageHeight"], sample["imageWidth"]), dtype=np.uint8)
            mask[fill_row_coords, fill_col_coords] = 255
            masks.append((int(shape["label"]), mask))
    return arr, masks

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        if K.sum(y_pred) == 0:
            return 0
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss
