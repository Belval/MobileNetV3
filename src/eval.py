import argparse

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

    return parser.parse_args()


def evaluate():
    """Runs inference with pretrained model.
    """

    args = parse_arguments()

    # Load model
    model = MobileNetV3LiteRASPP(shape=(448, 448, 3), n_class=args.class_count).build()
    model.load_weights(args.save_path, by_name=True)

    # Load image
    images = resize_and_crop(Image.open(args.input_image), 448)[np.newaxis, :, :, :]

    # Reference image
    Image.fromarray(resize_and_crop(Image.open(args.input_image), 56)).save("ref.jpg")

    predictions = model.predict(images, batch_size=1)

    predictions *= 255

    for i in range(predictions.shape[-1]):
        Image.fromarray(predictions[0, :, :, i]).convert('L').save(f"out/{i}.jpg")

if __name__ == "__main__":
    evaluate()
