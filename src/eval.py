import argparse
import os
import cv2
import numpy as np
from PIL import Image
from model import MobileNetV3LiteRASPP
from utils import resize_and_crop
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import jaccard_score
from data_generators import (
    isic_segmentation_data_generator,
)
from utils import(
    jaccard_distance
)

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
    parser.add_argument(
        "-pw",
        "--picture-width",
        type=int,
        nargs="?",
        help="Desired width of input image.",
        default=512,
    )
    parser.add_argument(
        "-ph",
        "--picture-height",
        type=int,
        nargs="?",
        help="Desired height of input image.",
        default=512,
    )

    return parser.parse_args()


def evaluate():
    """Runs inference with pretrained model.
    """

    args = parse_arguments()

    # Load model
    model = MobileNetV3LiteRASPP(shape=(args.picture_width, args.picture_height, 3), n_class=args.class_count, task=args.task)

    if(args.model_size == "large"):
        model = model.build_large()
    else:
        model = model.build_small()


    model.load_weights(args.save_path, by_name=True)
    acc = 0
    nb_not = 0

    for image_filename in os.listdir("../data/isic/test/imgs"):
        img = os.path.join("/home/guillaume/Documents/Git/MobileNetV3/data/isic/test/imgs", image_filename)
        mask = os.path.join("/home/guillaume/Documents/Git/MobileNetV3/data/isic/test/masks", f"{image_filename[:-4]}_segmentation.png")
        
        images = resize_and_crop(Image.open(img), 512)[np.newaxis, :, :, :]
        image = Image.open(img)
        image.thumbnail((512, 512))
        image_arr = np.array(image)
        images = np.zeros((1, 512, 512, 3), dtype=np.uint8)
        images[0, 0:image_arr.shape[0], 0:image_arr.shape[1], :] = image_arr

        pred = model.predict(images, batch_size=1)

        mask = np.array(resize_and_crop(Image.open(mask), 512, rgb=False))

        result = jaccard_score(mask.flatten().round(), pred[0].flatten().round())
        print(result)
        
        acc += result if result >= 0.65 else 0
        nb_not += 1 if result < 0.65 else 0

    print(nb_not)
    print(acc / 159)
    

    # Load image
    

    # print(predictions)

    # predictions *= 255
    # predictions = predictions.astype(np.uint8)


    # for i in range(imgs.shape[0]):
    #     im = imgs[i] * 255
    #     cv2.imwrite(f"out/imgs/{i}.png", im)

if __name__ == "__main__":
    evaluate()
