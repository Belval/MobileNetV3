import argparse
import os

import cv2
import math
import numpy as np
from PIL import Image
from model import MobileNetV3LiteRASPP
from utils import resize_and_crop


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
        "--input-image",
        type=str,
        nargs="?",
        help="Input image to run inference on",
    )
    parser.add_argument(
        "-d",
        "--input-directory",
        type=str,
        nargs="?",
        help="Input directory to run inference on every image",
    )
    parser.add_argument("--crop", default=False, action="store_true")
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        nargs="?",
        help="Output directory for the results",
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
        "-ms", "--model-size", type=str, nargs="?", help="Model size", default="large"
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        nargs="?",
        help="Batch size when generating segmentation masks",
        default=20,
    )

    return parser.parse_args()


def predict_images(paths, model, output_directory):
    images = np.zeros((len(paths), 512, 512, 3))

    orig_sizes = []
    for i, p in enumerate(paths):
        img = Image.open(p)
        orig_sizes.append(img.size)
        images[i, :, :, :] = resize_and_crop(img, 512)[np.newaxis, :, :, :]

    predictions = model.predict(images, batch_size=len(paths))
    predictions *= 255
    predictions = predictions.astype(np.uint8)

    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[-1]):
            new_w, new_h = (
                (int(orig_sizes[i][0] * 512 / orig_sizes[i][1]), 512)
                if orig_sizes[i][0] > orig_sizes[i][1]
                else (512, int(orig_sizes[i][1] * 512 / orig_sizes[i][0]))
            )
            img = Image.new("RGB", (new_w, new_h))
            ret, th = cv2.threshold(
                predictions[i, 0 : images[0].shape[1], 0 : images[0].shape[0], j],
                0,
                255,
                cv2.THRESH_OTSU,
            )
            img.paste(
                Image.fromarray(th).resize((512, 512)),
                (int(img.size[0] / 2 - 256), int(img.size[1] / 2 - 256)),
            )
            img = img.resize(orig_sizes[i])
            p = ".".join(os.path.basename(paths[i]).split(".")[0:-1])
            img.save(f"{output_directory}/{p}_{j}_segmentation.png")


def evaluate():
    """Runs inference with pretrained model.
    """

    args = parse_arguments()

    # Load model
    model = MobileNetV3LiteRASPP(
        shape=(512, 512, 3), n_class=args.class_count, task="segmentation"
    )

    if args.model_size == "large":
        model = model.build_large()
    else:
        model = model.build_small()

    model.load_weights(args.save_path, by_name=True)

    # Load image
    if args.input_image is not None:
        predict_images([args.input_image], model, args.output_directory)
    else:
        images = [
            os.path.join(args.input_directory, fp)
            for fp in os.listdir(args.input_directory)
        ]
        for i in range(int(math.ceil(len(images) / args.batch_size))):
            predict_images(
                images[i * args.batch_size : (i + 1) * args.batch_size],
                model,
                args.output_directory,
            )


if __name__ == "__main__":
    evaluate()
