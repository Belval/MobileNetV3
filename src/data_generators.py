import os
import random

import numpy as np
from PIL import Image
from pycocotools.coco import COCO


def coco_data_generator(path, batch_size, class_count):
    coco = COCO(path)
    coco.createIndex()

    def generator(coco, path, batch_size, class_count):
        images_directory = os.path.join(os.path.dirname(path), "data")
        files = {
            int(f.split(".")[0]): os.path.join(images_directory, f)
            for f in os.listdir(images_directory)
        }

        images = np.zeros((batch_size, 448, 448, 3))
        labels = np.zeros((batch_size, 448 // 8, 448 // 8, class_count))
        count = 0
        for image_id, segmentation in coco.imgToAnns.items():
            images[count, :, :, :] = np.array(
                Image.open(files[image_id]).resize((448, 448)).convert("RGB")
            )
            for ann in segmentation:
                labels[count, :, :, ann["category_id"] - 1] += np.array(
                    Image.fromarray(coco.annToMask(ann)).resize((448 // 8, 448 // 8))
                )
            labels[count, labels[count, :, :, :] > 0] = 1
            count += 1
            if count == batch_size:
                yield images, labels
                images = np.zeros((batch_size, 448, 448, 3))
                labels = np.zeros((batch_size, 448 // 8, 448 // 8, class_count))
                count = 0

    return generator(coco, path, batch_size, class_count), len(coco.imgToAnns)
