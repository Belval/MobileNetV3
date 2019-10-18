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
            labels = np.zeros((batch_size, class_count, 448 // 8, 448 // 8), dtype=np.uint8)
            count = 0
            for image_id, segmentation in coco.imgToAnns.items():
                image = Image.open(files[image_id])
                images[count, :, :, :] = resize_and_crop(image, 448)
                for i, ann in enumerate(segmentation):
                    arr = coco.annToMask(ann)
                    image = Image.fromarray(arr)
                    resized = resize_and_crop(image, 448 // 8, rgb=False)
                    labels[count, ann["category_id"] - 1, :, :] += resized
                labels[count, labels[count, :, :, :] > 0] = 1
                count += 1
                if count == batch_size:
                    yield images, labels
                    images = np.zeros((batch_size, 448, 448, 3), dtype=np.uint8)
                    labels = np.zeros((batch_size, 448 // 8, 448 // 8, class_count), dtype=np.uint8)
                    count = 0

    return generator(coco, path, batch_size, class_count), len(coco.imgToAnns)

if __name__ == '__main__':
    train_generator, c1 = coco_data_generator(
        "data/coco/train/instances_train2017.json",
        batch_size=32,
        class_count=100,
    )

    for blah, bleh in train_generator:
        print('Blah')