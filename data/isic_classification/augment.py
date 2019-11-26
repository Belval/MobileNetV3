import csv
import os
import random
import uuid
import pickle
from multiprocessing import Pool
from collections import Counter

import numpy as np
import imgaug.augmenters as iaa
from PIL import Image


def rotate_save(img, flip, angle, label, new_label_dict, out_dir):
    filename = str(uuid.uuid4()) + ".png"
    new_label_dict[filename] = label
    if not flip:
        img.rotate(angle, expand=True).save(os.path.join(out_dir, filename))
    else:
        img.rotate(angle, expand=True).transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(out_dir, filename))

def process_image(image_filename, in_dir, out_dir, label_dict, count):
    new_label_dict = {}
    img = Image.open(os.path.join(in_dir, image_filename))
    config = [(False, 0), (False, 90), (False, 180), (False, 270), (True, 0), (True, 90), (True, 180), (True, 270)]
    while count > 0:
        flip, angle = config[(count - 1) % len(config)]
        rotate_save(img, flip, angle, label_dict[image_filename[:-4]], new_label_dict, out_dir)
        count -= 1
    return new_label_dict

def main(in_dir, out_dir, labels_file):
    label_dict = {}
    with open(labels_file, "r") as f:
        csvfile = csv.reader(f)
        # Skip column description
        next(csvfile)
        for row in csvfile:
            label_dict[row[0]] = row[1:].index("1.0")

    new_label_dict = {}

    counter = Counter(label_dict.values())
    desired_counts = {k:int(0.5 * (max(counter.values()) - n) + n) for k, n in counter.most_common()}

    print(counter)
    print(desired_counts)

    files = os.listdir(in_dir)
    random.shuffle(files)
    print(len(files))
    p = Pool(16)
    dicts = p.starmap(
        process_image,
        [
            (
                image_filename,
                in_dir,
                out_dir,
                label_dict,
                int(desired_counts[label_dict[image_filename[:-4]]] / counter[label_dict[image_filename[:-4]]])
            )
            for image_filename in files
        ]
    )
        
    combined_dict = {}
    for d in dicts:
        combined_dict.update(d)

    with open("label_dict.pkl", "wb") as f:
        pickle.dump(combined_dict, f)

if __name__ == "__main__":
    main("train/", "train_aug/", "ISIC2018_Task3_Training_GroundTruth.csv")