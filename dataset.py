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

def make_csv():
    df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))

    legacy_df = pd.DataFrame(columns=['Num'])

    for id in df['ImageId'].values.tolist():
        legacy_df.loc[id] = 0
    for id in df['ImageId'].values.tolist():
        legacy_df.loc[id] += 1

    idx = legacy_df[legacy_df['Num'] >= 2].index
    legacy_df2 = legacy_df.drop(idx)
    legacy_df2['ImageId'] = legacy_df2.index

    df2 = pd.merge(df, legacy_df2, on='ImageId', how='right')
    df2 = df2.drop(['EncodedPixels'], axis='columns')
    df2.to_csv('train3.csv', index=False)

    im_list = glob.glob(os.path.join('./train_images', '*.jpg'))
    id_list = []
    for i in im_list:
        id_list.append(i[15:])

    df2 = pd.read_csv('train3.csv')
    for id in id_list:
        if id not in list(df['ImageId']):
            new = {'ImageId': id, 'ClassId': 0, 'Num': 0}
            df2 = df2.append(new, ignore_index=True)
    df3 = df2.rename(columns={'Num': 'Defection'})
    df3.to_csv('training.csv', index=False)

def make_folder():
    df = pd.read_csv('training.csv')
    image_files = list(df['ImageId'])
    labels = list(df['Defection'])

    X_train, X_test, y_train, y_test = train_test_split(image_files, labels, test_size=TEST_SIZE,
                                                        random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE)

    for i, image_file in enumerate(X_train):
        src = os.path.join(DATASET_DIR, 'train_images', image_file)
        if y_train[i] == 0:
            dst = os.path.join('./train', '0', image_file)
        else:
            dst = os.path.join('./train', '1', image_file)
        copyfile(src, dst)

    for i, image_file in enumerate(X_val):
        src = os.path.join(DATASET_DIR, 'train_images', image_file)
        if y_val[i] == 0:
            dst = os.path.join('./val', '0', image_file)
        else:
            dst = os.path.join('./val', '1', image_file)
        copyfile(src, dst)

    for i, image_file in enumerate(X_test):
        src = os.path.join(DATASET_DIR, 'train_images', image_file)
        if y_test[i] == 0:
            dst = os.path.join('./test', '0', image_file)
        else:
            dst = os.path.join('./test', '1', image_file)
        copyfile(src, dst)


if __name__ == '__main__':
    make_csv()
    make_folder()