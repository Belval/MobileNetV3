import argparse

import numpy as np
from PIL import Image


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


def eval():
    """Runs inference with pretrained model.
    """

    args = parse_arguments()

    model = MobileNetV3LiteRASPP(shape=(448, 448, 3), n_class=args.class_count).build()

    model.load_weights(args.save_path, by_name=True)

    images = np.array(Image.open(files[image_id]).resize((448, 448)).convert("RGB"))[
        np.newaxis, :, :, :
    ]

    predictions = model.predict(images, batch_size=1)

    print(predictions)
