import pandas as pd
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import glob
from shutil import copyfile
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook

TEST_SIZE = 0.1
VAL_SIZE = 0.15
RANDOM_STATE = 1
DATASET_DIR = '.'


def make_folder():
    df = pd.read_csv('training.csv')
    image_files = list(df['ImageId'])
    labels = list(df['ClassId'])

    X_train, X_test, y_train, y_test = train_test_split(image_files, labels, test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE)

    for i, image_file in enumerate(X_train):
        src = os.path.join(DATASET_DIR, 'train_images', image_file)
        if y_train[i] == 0:
            dst = os.path.join('./multi/train', '0', image_file)
        elif y_train[i] == 1:
            dst = os.path.join('./multi/train', '1', image_file)
        elif y_train[i] == 2:
            dst = os.path.join('./multi/train', '2', image_file)
        elif y_train[i] == 3:
            dst = os.path.join('./multi/train', '3', image_file)
        else:
            dst = os.path.join('./multi/train', '4', image_file)
        copyfile(src, dst)

    for i, image_file in enumerate(X_val):
        src = os.path.join(DATASET_DIR, 'train_images', image_file)
        if y_val[i] == 0:
            dst = os.path.join('./multi/val', '0', image_file)
        elif y_val[i] == 1:
            dst = os.path.join('./multi/val', '1', image_file)
        elif y_val[i] == 2:
            dst = os.path.join('./multi/val', '2', image_file)
        elif y_val[i] == 3:
            dst = os.path.join('./multi/val', '3', image_file)
        else:
            dst = os.path.join('./multi/val', '4', image_file)
        copyfile(src, dst)

    for i, image_file in enumerate(X_test):
        src = os.path.join(DATASET_DIR, 'train_images', image_file)
        if y_test[i] == 0:
            dst = os.path.join('./multi/test', '0', image_file)
        elif y_test[i] == 1:
            dst = os.path.join('./multi/test', '1', image_file)
        elif y_test[i] == 2:
            dst = os.path.join('./multi/test', '2', image_file)
        elif y_test[i] == 3:
            dst = os.path.join('./multi/test', '3', image_file)
        else:
            dst = os.path.join('./multi/test', '4', image_file)
        copyfile(src, dst)


if __name__ == '__main__':
    make_folder()