import csv
import os
import random
import pickle
from collections import Counter

import numpy as np
from pathlib import Path
from PIL import Image
from pycocotools.coco import COCO
from utils import (
    resize_and_crop,
    parse_labelme_file
)

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


def hazmat_data_generator(samples_dir, batch_size, class_count):
    def generator(samples_dir, batch_size, class_count):
        files = os.listdir(samples_dir)
        while True:
            images = np.zeros((batch_size, 1024, 1024, 3), dtype=np.uint8)
            labels = np.zeros((batch_size, 1024 // 8, 1024 // 8, class_count), dtype=np.uint8)
            count = 0
            random.shuffle(files)
            for f in files:
                centering = (round(random.random(), 1), round(random.random(), 1))
                arr, masks = parse_labelme_file(os.path.join(samples_dir, f))
                images[count, :, :, :] = resize_and_crop(arr, 1024, centering=centering)
                for l, mask in masks:
                    labels[count, :, :, l] += resize_and_crop(mask, 1024 // 8, centering=centering, rgb=False)
                count += 1
                if count == batch_size:
                    labels[labels > 0] = 1
                    yield images, labels
                    images = np.zeros((batch_size, 1024, 1024, 3), dtype=np.uint8)
                    labels = np.zeros((batch_size, 1024 // 8, 1024 // 8, class_count), dtype=np.uint8)
                    count = 0
    
    return generator(samples_dir, batch_size, class_count), len(os.listdir(samples_dir))


def isic_segmentation_data_generator(images_dir, masks_dir, batch_size, class_count, picture_size, model_size='large'):
    def generator(images_dir, masks_dir, batch_size, class_count):
        while True:
            images = np.zeros((batch_size, picture_size, picture_size, 3), dtype=np.uint8)

            if model_size == 'large':
                labels = np.zeros((batch_size, picture_size // 8, picture_size // 8, class_count), dtype=np.uint8)
            else:
                labels = np.zeros((batch_size, picture_size // 16, picture_size // 16, class_count), dtype=np.uint8)

            count = 0
            for image_filename in os.listdir(images_dir):
                centering = (round(random.random(), 1), round(random.random(), 1))
                if image_filename[-3:] not in ('jpg', 'png'):
                    continue
                p = os.path.join(os.getcwd(), images_dir, image_filename)
                image = Image.open(p)
                images[count, :, :, :] = resize_and_crop(image, picture_size, centering=centering)

                mask = Image.open(os.path.join(
                    masks_dir,
                    f"{image_filename[:-4]}_segmentation.png")
                )

                if model_size == 'large':
                    labels[count, :, :, 0] = resize_and_crop(mask, picture_size // 8, centering=centering, rgb=False)
                else:
                    labels[count, :, :, 0] = resize_and_crop(mask, picture_size // 16, centering=centering, rgb=False)

                labels[count, labels[count, :, :, :] > 0] = 1
                count += 1
                if count == batch_size:
                    yield images, labels
                    images = np.zeros((batch_size, picture_size, picture_size, 3), dtype=np.uint8)

                    if model_size == 'large':
                        labels = np.zeros((batch_size, picture_size // 8, picture_size // 8, class_count))
                    else:
                        labels = np.zeros((batch_size, picture_size // 16, picture_size // 16, class_count))

                    count = 0
    return generator(images_dir, masks_dir, batch_size, class_count), len(os.listdir(images_dir))

def isic_classification_data_generator(images_dir, labels_file, batch_size, class_count):
    label_dict = {}
    with open(labels_file, "r") as f:
        csvfile = csv.reader(f)
        # Skip column description
        next(csvfile)
        for row in csvfile:
            print(row)
            label_dict[row[0]] = row[1:].index("1.0")

    counter = {i:0 for i in range(class_count)}
    for image_filename in os.listdir(images_dir):
        counter[label_dict[image_filename[:-4]]] += 1
    weights = {i:int(max(counter.values())/counter[i]) for i in range(class_count)}

    def generator(images_dir, labels_file, batch_size, class_count):
        while True:
            images = np.zeros((batch_size, 512, 512, 3), dtype=np.uint8)
            labels = np.zeros((batch_size, class_count), dtype=np.uint8)
            count = 0
            for image_filename in os.listdir(images_dir):
                centering = (round(random.random(), 1), round(random.random(), 1))
                if image_filename[-3:] not in ('jpg', 'png'):
                    continue
                p = os.path.join(images_dir, image_filename)
                image = Image.open(p)
                images[count, :, :, :] = resize_and_crop(image, 512, centering=centering)
                labels[count, label_dict[image_filename[:-4]]] = 1
                count += 1
                if count == batch_size:
                    yield images, labels
                    images = np.zeros((batch_size, 512, 512, 3), dtype=np.uint8)
                    labels = np.zeros((batch_size, class_count))
                    count = 0
    
    return generator(images_dir, labels_file, batch_size, class_count), len(os.listdir(images_dir)), weights

def isic_classification_augmented_data_generator(images_dir, labels_file, batch_size, class_count):
    label_dict = pickle.load(open(labels_file, "rb"))

    counter = {i:1 for i in range(class_count)}
    for image_filename in os.listdir(images_dir):
        counter[label_dict[image_filename]] += 1
    weights = {i:int(max(counter.values())/counter[i]) for i in range(class_count)}

    def generator(images_dir, labels_file, batch_size, class_count):
        while True:
            images = np.zeros((batch_size, 512, 512, 3), dtype=np.uint8)
            labels = np.zeros((batch_size, class_count), dtype=np.uint8)
            count = 0
            for image_filename in os.listdir(images_dir):
                centering = (round(random.random(), 1), round(random.random(), 1))
                if image_filename[-3:] not in ('jpg', 'png'):
                    continue
                p = os.path.join(images_dir, image_filename)
                image = Image.open(p)
                images[count, :, :, :] = resize_and_crop(image, 512, centering=centering)
                labels[count, label_dict[image_filename]] = 1
                count += 1
                if count == batch_size:
                    yield images, labels
                    images = np.zeros((batch_size, 512, 512, 3), dtype=np.uint8)
                    labels = np.zeros((batch_size, class_count))
                    count = 0
    
    return generator(images_dir, labels_file, batch_size, class_count), len(os.listdir(images_dir)), weights

def isic_mixed_data_generator(
    class_images_dir,
    labels_file,
    seg_images_dir,
    seg_mask_dir,
    batch_size,
    class_count,
    model_size='large',
    proportions=0.5
):
    label_dict = {}
    with open(labels_file, "r") as f:
        csvfile = csv.reader(f)
        # Skip column description
        next(csvfile)
        for row in csvfile:
            label_dict[row[0]] = row[1:].index("1.0")

    counter = {i:0 for i in range(class_count)}
    for image_filename in os.listdir(class_images_dir):
        counter[label_dict[image_filename[:-4]]] += 1
    weights = {i:int(max(counter.values())/counter[i]) for i in range(class_count)}

    def classification_generator():
        while True:
            for image_filename in os.listdir(class_images_dir):
                centering = (round(random.random(), 1), round(random.random(), 1))
                if image_filename[-3:] not in ('jpg', 'png'):
                    continue
                p = os.path.join(class_images_dir, image_filename)
                image = Image.open(p)
                yield (
                    resize_and_crop(image, 512, centering=centering),
                    label_dict[image_filename[:-4]]
                )

    def segmentation_generator():
        while True:
            for image_filename in os.listdir(seg_images_dir):
                centering = (round(random.random(), 1), round(random.random(), 1))
                if image_filename[-3:] not in ('jpg', 'png'):
                    continue
                p = os.path.join(os.getcwd(), seg_images_dir, image_filename)
                image = Image.open(p)
                mask = Image.open(os.path.join(
                    seg_mask_dir,
                    f"{image_filename[:-4]}_segmentation.png")
                )

                yield (
                    resize_and_crop(image, 512, centering=centering),
                    resize_and_crop(mask, 512 // (8 if model_size == "large" else 16), centering=centering, rgb=False)
                )

    def generator():
        while True:
            images = np.zeros((batch_size, 512, 512, 3), dtype=np.uint8)
            class_labels = np.zeros((batch_size, class_count), dtype=np.uint8)
            if model_size == 'large':
                seg_labels = np.zeros((batch_size, 512 // 8, 512 // 8, class_count))
            else:
                seg_labels = np.zeros((batch_size, 512 // 16, 512 // 16, class_count))

            class_generator = classification_generator()
            seg_generator = segmentation_generator()

            for i in range(batch_size):
                if random.random() < proportions:
                    arr, lbl = next(class_generator)
                    images[i, :, :, :] = arr
                    class_labels[i, lbl] = 1
                else:
                    arr, mask = next(seg_generator)
                    images[i, :, :, :] = arr
                    seg_labels[i, mask > 0] = 1

            yield images, {
                "segme_out": seg_labels,
                "class_out": class_labels,
            }

    return generator(), len(os.listdir(class_images_dir)) + len(os.listdir(seg_images_dir)), weights

def isic_mixed_augmented_data_generator(
    class_images_dir,
    labels_file,
    seg_images_dir,
    seg_mask_dir,
    batch_size,
    class_count,
    model_size='large',
    proportions=0.5
):
    label_dict = pickle.load(open(labels_file, "rb"))

    counter = {i:1 for i in range(class_count)}
    for image_filename in os.listdir(class_images_dir):
        counter[label_dict[image_filename]] += 1
    weights = {i:int(max(counter.values())/counter[i]) for i in range(class_count)}

    def classification_generator():
        while True:
            for image_filename in os.listdir(class_images_dir):
                centering = (round(random.random(), 1), round(random.random(), 1))
                if image_filename[-3:] not in ('jpg', 'png'):
                    continue
                p = os.path.join(class_images_dir, image_filename)
                image = Image.open(p)
                yield (
                    resize_and_crop(image, 512, centering=centering),
                    label_dict[image_filename]
                )

    def segmentation_generator():
        while True:
            for image_filename in os.listdir(seg_images_dir):
                centering = (round(random.random(), 1), round(random.random(), 1))
                if image_filename[-3:] not in ('jpg', 'png'):
                    continue
                p = os.path.join(os.getcwd(), seg_images_dir, image_filename)
                image = Image.open(p)
                mask = Image.open(os.path.join(
                    seg_mask_dir,
                    f"{image_filename[:-4]}_segmentation.png")
                )

                yield (
                    resize_and_crop(image, 512, centering=centering),
                    resize_and_crop(mask, 512 // (8 if model_size == "large" else 16), centering=centering, rgb=False)
                )

    def generator():
        while True:
            images = np.zeros((batch_size, 512, 512, 3), dtype=np.uint8)
            class_labels = np.zeros((batch_size, class_count), dtype=np.uint8)
            if model_size == 'large':
                seg_labels = np.zeros((batch_size, 512 // 8, 512 // 8, class_count))
            else:
                seg_labels = np.zeros((batch_size, 512 // 16, 512 // 16, class_count))

            class_generator = classification_generator()
            seg_generator = segmentation_generator()

            for i in range(batch_size):
                if random.random() < proportions:
                    arr, lbl = next(class_generator)
                    images[i, :, :, :] = arr
                    class_labels[i, lbl] = 1
                else:
                    arr, mask = next(seg_generator)
                    images[i, :, :, :] = arr
                    seg_labels[i, mask > 0] = 1

            yield images, {
                "segme_out": seg_labels,
                "class_out": class_labels,
            }

    return generator(), len(os.listdir(class_images_dir)) + len(os.listdir(seg_images_dir)), weights

if __name__ == '__main__':
    train_generator, c1 = isic_data_generator(
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