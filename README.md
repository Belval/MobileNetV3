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

## COCO categories

```
0 - 1 person
1 - 2 bicycle
2 - 3 car
3 - 4 motorcycle
4 - 5 airplane
5 - 6 bus
6 - 7 train
7 - 8 truck
8 - 9 boat
9 - 10 traffic light
10 - 11 fire hydrant
11 - 13 stop sign
12 - 14 parking meter
13 - 15 bench
14 - 16 bird
15 - 17 cat
16 - 18 dog
17 - 19 horse
18 - 20 sheep
19 - 21 cow
20 - 22 elephant
21 - 23 bear
22 - 24 zebra
23 - 25 giraffe
24 - 27 backpack
25 - 28 umbrella
26 - 31 handbag
27 - 32 tie
28 - 33 suitcase
29 - 34 frisbee
30 - 35 skis
31 - 36 snowboard
32 - 37 sports ball
33 - 38 kite
34 - 39 baseball bat
35 - 40 baseball glove
36 - 41 skateboard
37 - 42 surfboard
38 - 43 tennis racket
39 - 44 bottle
40 - 46 wine glass
41 - 47 cup
42 - 48 fork
43 - 49 knife
44 - 50 spoon
45 - 51 bowl
46 - 52 banana
47 - 53 apple
48 - 54 sandwich
49 - 55 orange
50 - 56 broccoli
51 - 57 carrot
52 - 58 hot dog
53 - 59 pizza
54 - 60 donut
55 - 61 cake
56 - 62 chair
57 - 63 couch
58 - 64 potted plant
59 - 65 bed
60 - 67 dining table
61 - 70 toilet
62 - 72 tv
63 - 73 laptop
64 - 74 mouse
65 - 75 remote
66 - 76 keyboard
67 - 77 cell phone
68 - 78 microwave
69 - 79 oven
70 - 80 toaster
71 - 81 sink
72 - 82 refrigerator
73 - 84 book
74 - 85 clock
75 - 86 vase
76 - 87 scissors
77 - 88 teddy bear
78 - 89 hair drier
79 - 90 toothbrush
```