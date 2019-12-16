import argparse
import os
import csv
import cv2
import numpy as np
from PIL import Image
from model import MobileNetV3LiteRASPP
from utils import resize_and_crop
from sklearn import metrics

def parse_arguments():
    """Parse commandline arguments
    """

    parser = argparse.ArgumentParser(description="Train the MobileNetV3 model")

    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        nargs="?",
        help="Path where the model is saved",
        default="out/",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        nargs="?",
        help="Input directory to run inference on",
    )
    parser.add_argument(
        "-l",
        "--labels-file",
        type=str,
        nargs="?",
        help="Label file",
    )
    parser.add_argument(
        "-cc",
        "--class-count",
        type=int,
        nargs="?",
        help="Number of classes",
        default=90,  # Number of classes in coco 2017
    )
    parser.add_argument(
        "-ms",
        "--model-size",
        type=str,
        nargs="?",
        help="Model size",
        default="large", 
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        nargs="?",
        help="Task",
        default="classification", 
    )

    return parser.parse_args()


def evaluate():
    """Runs inference with pretrained model.
    """

    args = parse_arguments()

    # Load model
    model = MobileNetV3LiteRASPP(shape=(512, 512, 3), n_class=args.class_count, task=args.task)

    label_dict = {}
    with open(args.labels_file, "r") as f:
        csvfile = csv.reader(f)
        # Skip column description
        next(csvfile)
        for row in csvfile:
            label_dict[row[0]] = row[1:].index("1.0")

    if(args.model_size == "large"):
        model = model.build_large()
    else:
        model = model.build_small()

    model.load_weights(args.save_path, by_name=True)

    images = np.zeros((len(os.listdir(args.input_dir)), 512, 512, 3))
    labels = np.zeros((len(os.listdir(args.input_dir)), args.class_count))
    for i, filename in enumerate(os.listdir(args.input_dir)):
        img = resize_and_crop(Image.open(os.path.join(args.input_dir, filename)), 512)
        images[i, :, :, :] = np.array(img)
        labels[i, label_dict[filename[:-4]]] = 1

    preds = model.predict(images)
    print(labels.argmax(axis=1))
    print(preds.argmax(axis=1))
    matrix = metrics.confusion_matrix(labels.argmax(axis=1), preds.argmax(axis=1))
    f1 = metrics.f1_score(labels.argmax(axis=1), preds.argmax(axis=1), average=None)
    acc = 0
    for i in range(matrix.shape[0]):
        acc += matrix[i, i] / sum(matrix[i, :])
    print(np.around(matrix / np.sum(matrix, axis=1)[:, None], decimals=2))
    print(acc / args.class_count)
    print(f1)

if __name__ == "__main__":
    evaluate()
