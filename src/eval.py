import argparse

import cv2
import numpy as np
from PIL import Image
from model import MobileNetV3LiteRASPP
from utils import resize_and_crop

def parse_arguments():
    """Parse commandline arguments
    """

    parser = argparse.ArgumentParser(description="Train the MobileNetV3 model")

    parser.add_argument(
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
        "-cc",
        "--class-count",
        type=int,
        nargs="?",
        help="Number of classes",
        default=90,  # Number of classes in coco 2017
    )
    parser.add_argument(
        "-th",
        "--threshold",
        type=float,
        nargs="?",
        help="Threshold to consider a pixel as belonging to a category",
        default=0.5
    )

    return parser.parse_args()


def evaluate():
    """Runs inference with pretrained model.
    """

    args = parse_arguments()

    # Load model
    model = MobileNetV3LiteRASPP(shape=(1024, 1024, 3), n_class=args.class_count).build()
    model.load_weights(args.save_path, by_name=True)

    # Load image
    #images = resize_and_crop(Image.open(args.input_image), 1024)[np.newaxis, :, :, :]
    image = Image.open(args.input_image)
    image.thumbnail((1024, 1024))
    image_arr = np.array(image)
    images = np.zeros((1, 1024, 1024, 3), dtype=np.uint8)
    images[0, 0:image_arr.shape[0], 0:image_arr.shape[1], :] = image_arr

    # Reference image
    image.thumbnail((128, 128))
    image.save("ref.png")

    predictions = model.predict(images, batch_size=1)

    print(predictions)

    predictions *= 255
    predictions = predictions.astype(np.uint8)


    for i in range(predictions.shape[-1]):
        ret, th = cv2.threshold(predictions[0, 0:image.size[1], 0:image.size[0], i], 0, 255, cv2.THRESH_OTSU)
        cv2.imwrite(f"out/{i}.png", th)

if __name__ == "__main__":
    evaluate()
