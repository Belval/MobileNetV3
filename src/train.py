import argparse
import errno
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from loss import dice_coef_multilabel_builder
import pandas as pd
from model import MobileNetV3LiteRASPP
from data_generators import (
    coco_data_generator,
    hazmat_data_generator,
    #hltid_data_generator,
    isic_segmentation_data_generator,
    isic_classification_data_generator,
)


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
        default=50,
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        nargs="?",
        help="Training batch size",
        default=6,
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        nargs="?",
        help="Learning rate",
        default=0.0003,
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

    model = MobileNetV3LiteRASPP(
        shape=(1024, 1024, 3),
        n_class=args.class_count,
        task="classification",
    ).build()

    try:
        os.mkdir(args.save_path)
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise
        pass

    early_stop = EarlyStopping(monitor="val_acc", patience=5, mode="auto")

    model.compile(
        #loss=dice_coef_multilabel_builder(args.class_count),
        loss=categorical_crossentropy,
        optimizer=Adam(lr=args.learning_rate),
        metrics=["accuracy"]
    )

    #train_generator, c1 = coco_data_generator(
    #    "../data/coco/train/instances_train2017.json",
    #    batch_size=args.batch_size,
    #    class_count=args.class_count,
    #)
    #val_generator, c2 = coco_data_generator(
    #    "../data/coco/val/instances_val2017.json",
    #    batch_size=args.batch_size,
    #    class_count=args.class_count,
    #)

    #train_generator, c1 = isic_segmentation_data_generator(
    #    "../data/isic_segmentation/train/imgs",
    #    "../data/isic_segmentation/train/masks",
    #    batch_size=args.batch_size,
    #    class_count=args.class_count,
    #)
    #val_generator, c2 = isic_segmentation_data_generator(
    #    "../data/isic_segmentation/val/imgs",
    #    "../data/isic_segmentation/val/masks",
    #    batch_size=args.batch_size,
    #    class_count=args.class_count,
    #)

    train_generator, c1 = isic_classification_data_generator(
        "../data/isic_classification/train/",
        "../data/isic_classification/ISIC2018_Task3_Training_GroundTruth.csv",
        batch_size=args.batch_size,
        class_count=args.class_count,
    )
    val_generator, c2 = isic_classification_data_generator(
        "../data/isic_classification/val/",
        "../data/isic_classification/ISIC2018_Task3_Training_GroundTruth.csv",
        batch_size=args.batch_size,
        class_count=args.class_count,
    )

    #train_generator, c1 = hazmat_data_generator(
    #    "../data/hazmat/train/",
    #    batch_size=args.batch_size,
    #    class_count=args.class_count,
    #)
    #val_generator, c2 = hazmat_data_generator(
    #    "../data/hazmat/val/",
    #    batch_size=args.batch_size,
    #    class_count=args.class_count,
    #)

    #train_generator, c1 = hltid_data_generator(
    #    "../data/HLTID/train/",
    #    batch_size=args.batch_size,
    #    class_count=args.class_count,
    #)
    #val_generator, c2 = hltid_data_generator(
    #    "../data/HLTID/val/",
    #    batch_size=args.batch_size,
    #    class_count=args.class_count,
    #)

    hist = model.fit_generator(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=c1 // args.batch_size,
        validation_steps=c2 // args.batch_size,
        epochs=args.iteration_count,
        callbacks=[early_stop],
    )

    try:
        df = pd.DataFrame.from_dict(hist.history)
        df.to_csv(os.path.join(args.save_path, "hist.csv"), encoding="utf-8", index=False)
    except Exception as ex:
        print(f"Unable to save histogram: {str(ex)}")

    model.save_weights(os.path.join(args.save_path, f"{int(time.time())}_weights.h5"))


if __name__ == "__main__":
    train()
