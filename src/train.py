import argparse
import errno
import os
import time

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from loss import ce_dice_loss
import pandas as pd
from model import MobileNetV3LiteRASPP
from data_generators import coco_data_generator


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
        default=32,
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        nargs="?",
        help="Learning rate",
        default=0.001,
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


def train():
    """Train MobileNetV3
    """

    args = parse_arguments()

    model = MobileNetV3LiteRASPP(shape=(448, 448, 3), n_class=args.class_count).build()

    try:
        os.mkdir(args.save_path)
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise
        pass

    early_stop = EarlyStopping(monitor="val_acc", patience=5, mode="auto")
    model.compile(
        loss=ce_dice_loss, optimizer=Adam(lr=args.learning_rate), metrics=["accuracy"]
    )

    train_generator, c1 = coco_data_generator(
        "../data/coco/train/instances_train2017.json",
        batch_size=args.batch_size,
        class_count=args.class_count,
    )
    val_generator, c2 = coco_data_generator(
        "../data/coco/val/instances_val2017.json",
        batch_size=args.batch_size,
        class_count=args.class_count,
    )

    print(c1)
    print(c2)

    hist = model.fit_generator(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=c1 // args.batch_size,
        validation_steps=c2 // args.batch_size,
        epochs=args.iteration_count,
        callbacks=[early_stop],
    )

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv(os.path.join(args.model_path, "hist.csv"), encoding="utf-8", index=False)
    model.save_weights(os.path.join(args.model_path, f"{int(time.time())}_weights.h5"))


if __name__ == "__main__":
    train()
