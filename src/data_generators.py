import os
import random

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from utils import resize_and_crop

def coco_data_generator(path, batch_size, class_count):
    coco = COCO(path)
    coco.createIndex()

    def generator(coco, path, batch_size, class_count):
        images_directory = os.path.join(os.path.dirname(path), "data")
        files = {
            int(f.split(".")[0]): os.path.join(images_directory, f)
            for f in os.listdir(images_directory)
        }

        while True:
            images = np.zeros((batch_size, 448, 448, 3), dtype=np.uint8)
            labels = np.zeros((batch_size, 448 // 8, 448 // 8, class_count), dtype=np.uint8)
            count = 0
            for image_id, segmentation in coco.imgToAnns.items():
                image = Image.open(files[image_id])
                images[count, :, :, :] = resize_and_crop(image, 448)

                for ann in segmentation:
                    image = Image.fromarray(coco.annToMask(ann))
                    labels[count, :, :, ann["category_id"] - 1] += resize_and_crop(image, 448 // 8, rgb=False)
                labels[count, labels[count, :, :, :] > 0] = 1
                count += 1
                if count == batch_size:
                    yield images, labels
                    images = np.zeros((batch_size, 448, 448, 3), dtype=np.uint8)
                    labels = np.zeros((batch_size, 448 // 8, 448 // 8, class_count), dtype=np.uint8)
                    count = 0

    return generator(coco, path, batch_size, class_count), len(coco.imgToAnns)
