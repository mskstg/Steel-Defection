import numpy as np # linear algebra
import pandas as pd
import os
import cv2
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm

train_df = pd.read_csv("./train.csv")
palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]

col = 7096
PATH = './train_images/'
name = train_df.iloc[col]['ImageId']


def show_mask_image(col):
    name = train_df.iloc[col]['ImageId']
    label = train_df.iloc[col]['EncodedPixels']
    cls = np.int(train_df.iloc[col]['ClassId'])

    if cls != 0:
        mask_label = np.zeros(1600 * 256, dtype=np.uint8)
        label = label.split(" ")
        positions = map(int, label[0::2])
        length = map(int, label[1::2])
        for pos, le in zip(positions, length):
            mask_label[pos - 1:pos + le - 1] = 1
        mask = mask_label.reshape(256, 1600, order='F')
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    img = cv2.imread(PATH + name)
    fig, ax = plt.subplots(figsize=(15, 15))

    if cls != 0:
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[cls-1], 2)
    ax.set_title(str(cls))
    ax.imshow(img)
    plt.show()


if __name__ == '__main__':
    show_mask_image(col)



