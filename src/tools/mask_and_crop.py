import os
import sys

import numpy as np
from PIL import Image

def get_bbox(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox

def main():
    images_dir, masks_dir, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    for f in os.listdir(images_dir):
        img_p = os.path.join(images_dir, f)
        mask_p = os.path.join(masks_dir, ".".join(f.split('.')[:-1]) + "_segmentation.png")
        try:
            top, bottom, left, right = get_bbox(np.array(Image.open(mask_p).convert('1')))
            cropped_img = Image.fromarray(np.array(Image.open(img_p))[top:bottom, left:right, :])
            cropped_img.save(os.path.join(output_dir, f))
        except:
            Image.open(img_p).save(os.path.join(output_dir, f))

if __name__ == "__main__":
    main()