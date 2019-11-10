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
            images = np.zeros((batch_size, 1024, 1024, 3), dtype=np.uint8)
            labels = np.zeros((batch_size, 1024 // 8, 1024 // 8, class_count), dtype=np.uint8)
            count = 0
            for image_id, segmentation in coco.imgToAnns.items():
                image = Image.open(files[image_id])
                images[count, :, :, :] = resize_and_crop(image, 1024)
                for i, ann in enumerate(segmentation):
                    arr = coco.annToMask(ann)
                    image = Image.fromarray(arr)
                    resized = resize_and_crop(image, 1024 // 8, rgb=False)
                    labels[count, :, :, ann["category_id"] - 1] += resized
                labels[count, labels[count, :, :, :] > 0] = 1
                count += 1
                if count == batch_size:
                    yield images, labels
                    images = np.zeros((batch_size, 1024, 1024, 3), dtype=np.uint8)
                    labels = np.zeros((batch_size, 1024 // 8, 1024 // 8, class_count), dtype=np.uint8)
                    count = 0

    return generator(coco, path, batch_size, class_count), len(coco.imgToAnns)

def mask_data_generator(images_dir, masks_dir, batch_size, class_count):
    def generator(images_dir, masks_dir, batch_size, class_count):
        while True:
            images = np.zeros((batch_size, 1024, 1024, 3), dtype=np.uint8)
            labels = np.zeros((batch_size, 1024 // 8, 1024 // 8, class_count), dtype=np.uint8)
            count = 0
            for image_filename in os.listdir(images_dir):
                if image_filename[-3:] not in ('jpg', 'png'):
                    continue
                p = os.path.join(images_dir, image_filename)
                image = Image.open(p)
                images[count, :, :, :] = resize_and_crop(image, 1024)
                mask = Image.open(os.path.join(
                    masks_dir,
                    f"{image_filename[:-4]}_segmentation.png")
                )
                labels[count, :, :, 0] = resize_and_crop(mask, 1024 // 8, rgb=False)
                labels[count, labels[count, :, :, :] > 0] = 1
                count += 1
                if count == batch_size:
                    yield images, labels
                    images = np.zeros((batch_size, 1024, 1024, 3), dtype=np.uint8)
                    labels = np.zeros((batch_size, 1024 // 8, 1024 // 8, class_count))
                    count = 0
    
    return generator(images_dir, masks_dir, batch_size, class_count), len(os.listdir(images_dir))


if __name__ == '__main__':
    train_generator, c1 = mask_data_generator(
        "../data/isic/imgs",
        "../data/isic/masks",
        batch_size=32,
        class_count=1,
    )

    for blah, bleh in train_generator:
        Image.fromarray(blah[0]).save('bloh.png')
        Image.fromarray(np.squeeze(bleh[0] * 255)).save('bloh_label.png')
        print('Blah')
        input()