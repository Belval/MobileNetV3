import base64
import json
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
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

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
