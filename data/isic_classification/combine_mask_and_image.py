import os
import sys

import numpy as np
from PIL import Image

def main():
    images_dir, masks_dir, output_dir = sys.argv[1], sys.argv[2], sys.argv[3]

    for f in os.listdir(images_dir):
        img_p = os.path.join(images_dir, f)
        mask_p = os.path.join(masks_dir, ".".join(f.split('.')[:-1]) + "_segmentation.png")
        image = Image.open(img_p)
        mask = Image.open(mask_p).convert('1')
        arr = np.array(image)
        arr[np.array(mask) == 0] = 0
        image = Image.fromarray(arr)
        image.save(os.path.join(output_dir, f))

if __name__=='__main__':
    main()