import argparse

from .model import MobileNetV3

def parse_arguments():
    """Parse commandline arguments
    """

    parser = argparse.ArgumentParser(description="Train the MobileNetV3 model")

    parser.add_argument(
        "--save-path",
        type=str,
        nargs="?",
        help="Path where the model will be saved",
        default="out/",
    )
    parser.add_argument(
        "-it",
        "--iteration-count",
        type=int,
        nargs="?",
        help="Training iteration count",
        default=10000,
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        nargs="?",
        help="Training batch size",
        default=128,
    )

    return parser.parse_args()

def train():
    """Train MobileNetV3
    """

    args = parse_arguments()

    model = MobileNetV3(shape=(-1, 224, 224, 3)).build()



if __name__ == '__main__':
    train()