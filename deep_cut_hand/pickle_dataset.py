#!/usr/bin/python3
import cv2
import numpy as np
import os
import pandas as pd
import h5py
import sys

# Use N images of dataset, If it is -1 using all dataset
CUT_DATASET = -1

# Image resize
# IMAGE_SIZE = (299, 299)
# IMAGE_SIZE = (212, 212)
IMAGE_SIZE = (224, 224)
IMAGE_SIZE = (500, 500)

# Turn saving renders feature on/off
SAVE_RENDERS = False

# Create intermediate images in separate folders for debugger.
# mask, cut_hand, delete_object, render
SAVE_IMAGE_FOR_DEBUGGER = False

# For this problem the validation and test data provided by the concerned authority did not have labels,
# so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
train_dir = os.path.join(__location__, TRAIN_DIR)


def processImage(img_path):
    # Read a image
    img = cv2.imread(img_path, 0)

    # TODO completar espacion para hacerla cuadrada

    # ====================== show the images ================================
    if SAVE_IMAGE_FOR_DEBUGGER or SAVE_RENDERS:
        cv2.imwrite(os.path.join(__location__, TRAIN_DIR, "render", img_file), img)

    # Resize the images
    img = cv2.resize(img, IMAGE_SIZE)
    # Return to original colors
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Convert the image into an 8 bit array
    return np.asarray(img, dtype=np.float32)


# Write hdf5 file
def writeFile(gender, dataset, X_train, x_gender, y_age):
    print("Saving", gender, dataset, "data...")
    file_name = gender + "-" + dataset + "-" + ".hdf5"

    path_to_save = os.path.join(__location__, "packaging-dataset")
    if GENERATE_IMAGE_FOR_AUTOENCODER:
        path_to_save = os.path.join(path_to_save, "for_autoencoder")

    with h5py.File(os.path.join(path_to_save, file_name), "w") as f:
        f.create_dataset(
            "img",
            data=X_train,
            dtype=np.float32,
            compression="gzip",
            compression_opts=5,
        )
        f.create_dataset("age", data=y_age, dtype=np.uint8)
        f.create_dataset("gender", data=x_gender, dtype=np.uint8)
        f.close()


#!/usr/bin/python3
from multiprocessing import Process
from utilities import Console, updateProgress, getHistogram
from makeHandsFromCSV import makeHandsHuman
import cv2
import fnmatch
import h5py
import multiprocessing
import numpy as np
import os
import platform

# Use N images of dataset, If it is -1 using all dataset
CUT_DATASET = 1000
# Get lower and upper for csv and make hands img
MAKE_HANDS_FROM_HUMAN = True

# For this problem the validation and test data provided by the concerned authority did not have labels,
# so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def saveDataSet(X_train, y_train):
    Console.info("Save dataset")
    file_path = os.path.join(
        __location__, "dataset_hands", "histogram-hand-dataset.hdf5"
    )
    with h5py.File(file_path, "w") as f:
        f.create_dataset("hist", data=X_train)
        f.create_dataset("valid", data=y_train)
        f.flush()
        f.close()


def loadDataSet(path, files=[], hands_valid=1):
    X_train = []
    y_train = []

    total_file = len(files)
    for i in range(total_file):
        img_file = files[i]
        # Update the progress bar
        progress = float(i / total_file), (i + 1)
        updateProgress(progress[0], progress[1], total_file, img_file)

        # Get image's path
        img_path = os.path.join(path, img_file)
        # Read a image
