# MobileNetV3

A tensorflow implementation of the paper "Searching for MobileNetV3" with a R-ASPP segmenter head.

## Installation

`pip install -r requirements.txt`

## Results

TODO

## Train from scratch

Currently, the only available training option is with COCO 2017 instances challenge. You can download the files [here](http://cocodataset.org/#download).

Once that's done, create this hierarchy in your directories:

![Directories](images/directories.png)

You should be able to run it with `python3 train.py`

## Evaluation

Have an image ready to be evaluated.

Run `python3 eval.py --model-path out/ -i your_image.png`
