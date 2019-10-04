import numpy as np
from PIL import Image

def resize_and_crop(img, min_side, rgb=True):
    w, h = img.size
    if h < w:
        img = img.resize((int(w * (min_side / h)) + 1, min_side))
    else:
        img = img.resize((min_side, int(h * (min_side / w)) + 1))
    
    if rgb:
        img = img.convert("RGB")
        return np.array(img)[0:min_side, 0:min_side, :]

    return np.array(img)[0:min_side, 0:min_side]