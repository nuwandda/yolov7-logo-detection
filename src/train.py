import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split

random.seed(108)

# Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        file_class = f.split('/')[3]
        file_name = f.split('/')[-1]
        file_name = file_class + '_' + file_name
        try:
            shutil.copy(f, destination_folder + '/' + file_name)
        except:
            print(f)
            assert False


def get_annotations(path):
    annotations = []
    images = []
    for folder in glob(path + '/*/', recursive = True):
        for subfolder in glob(folder + '/*/', recursive = True):
            for txt in glob(subfolder + '*.txt'):
                annotations.append(txt)
                jpg = txt.replace('txt', 'jpg')
                images.append(jpg)

    return annotations, images


def main():
    annotations, images = get_annotations('data/LogoDet-3K')

    # Split the dataset into train-valid-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

    os.system('mkdir data/images data/annotations')
    os.system('mkdir data/images/train data/images/val data/images/test data/annotations/train data/annotations/val data/annotations/test')

    # Move the splits into their folders
    move_files_to_folder(train_images, 'data/images/train')
    move_files_to_folder(val_images, 'data/images/val/')
    move_files_to_folder(test_images, 'data/images/test/')
    move_files_to_folder(train_annotations, 'data/annotations/train/')
    move_files_to_folder(val_annotations, 'data/annotations/val/')
    move_files_to_folder(test_annotations, 'data/annotations/test/')

    os.system('cp -r data/annotations data/labels')


if __name__ == "__main__":
    main()
