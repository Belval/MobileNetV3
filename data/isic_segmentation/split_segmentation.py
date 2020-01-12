import os
import random
import sys

def mkdir_and_transfer(imgs, in_dir, out_dir):
    try:
        os.makedirs(os.path.join(out_dir, "imgs"))
    except:
        pass

    try:
        os.makedirs(os.path.join(out_dir, "masks"))
    except:
        pass

    for f in imgs:
        os.rename(os.path.join(in_dir, "imgs", f), os.path.join(out_dir, "imgs", f))
        os.rename(os.path.join(in_dir, "masks", f[:-4] + "_segmentation.png"), os.path.join(out_dir, "masks", f[:-4] + "_segmentation.png"))

def main(args):
    in_dir = args[1]
    val_prop = float(args[2])
    test_prop = float(args[3])

    imgs = [f for f in os.listdir(os.path.join(in_dir, "imgs"))]

    random.shuffle(imgs)

    train = imgs[0:int(len(imgs) * (1 - val_prop - test_prop))]
    val = imgs[int(len(imgs) * (1 - val_prop - test_prop)):int(len(imgs) * (1 - test_prop))]
    test = imgs[int(len(imgs) * (1 - test_prop)):]

    mkdir_and_transfer(val, in_dir, "val")
    mkdir_and_transfer(test, in_dir, "test")

if __name__ == '__main__':
    main(sys.argv)
