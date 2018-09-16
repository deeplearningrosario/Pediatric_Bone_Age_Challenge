#!/usr/bin/python3
#
# Data set for auto-encoder
#
from makeHandsFromCSV import makeHandsHuman
from multiprocessing import Process
from utilities import Console, updateProgress, getHistogram
import cv2
import fnmatch
import h5py
import multiprocessing
import numpy as np
import os
import platform

# Image resize
# IMAGE_SIZE = (212, 212)
IMAGE_SIZE = (224, 224)
# IMAGE_SIZE = (299, 299)
# IMAGE_SIZE = (500, 500)

# Turn saving renders feature on/off
SAVE_RENDERS = False

# Create intermediate images in separate folders for debugger.
# mask, cut_hand, delete_object, render
SAVE_IMAGE_FOR_DEBUGGER = False

# Get lower and upper for csv and make hands img
MAKE_HANDS_FROM_HUMAN = False

# For this problem the validation and test data provided by the concerned authority did not have labels,
# so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Write hdf5 file
def writeFile(dataset, X_img, y_img):
    Console.log("Saving", dataset, "data...")
    file_name = "img-for-autoencoder-" + dataset + ".hdf5"
    path_to_save = os.path.join(__location__, "dataset", file_name)

    with h5py.File(os.path.join(path_to_save), "w") as f:
        f.create_dataset(
            "x_img",
            data=X_img,
            dtype=np.float16,
            compression="gzip",
            compression_opts=5,
        )
        f.create_dataset("y_img", data=y_img, dtype=np.uint8)
        f.close()


def saveDataSet(X_img, y_img):
    Console.info("Save dataset")
    X_img = np.asarray(X_img, dtype=np.float16)
    y_img = np.asarray(y_img, dtype=np.float16)
    # Split images dataset
    k = int(len(X_img) / 6)
    writeFile("testing", X_img[:k, :, :, :], y_img[:k, :, :, :])
    writeFile("validation", X_img[k : 2 * k, :, :, :], y_img[k : 2 * k, :, :, :])
    writeFile("training", X_img[2 * k :, :, :, :], y_img[2 * k :, :, :, :])


def processeImg(img_path):
    img = cv2.imread(img_path)  # Read a image
    img = cv2.resize(img, IMAGE_SIZE)  # Resize the images
    return img


def loadDataSet(files=[]):
    total_file = len(files)

    # defined path
    path = os.path.join(__location__, "dataset")
    path_original = os.path.join(path, "original")
    path_hands = os.path.join(path, "hands")

    X_img = []
    y_img = []

    for i in range(total_file):
        img_file = files[i]
        # Update the progress bar
        progress = float(i / total_file), (i + 1)
        updateProgress(progress[0], progress[1], total_file, img_file)

        # Get image's path
        img_path = os.path.join(path_original, img_file)
        X_img.append(processeImg(img_path))

        img_path = os.path.join(path_hands, img_file)
        y_img.append(processeImg(img_path))

    return X_img, y_img


# list all the image files and randomly unravel them,
# in each case you take the first N from the unordered list
def getFiles():
    Console.info("Reading img...")
    rta = []
    # defined path
    path = os.path.join(__location__, "dataset")
    path_original = os.path.join(path, "original")
    path_hands = os.path.join(path, "hands")
    # file names on train_dir
    files_original = os.listdir(path_original)
    files_hand = os.listdir(path_hands)
    # filter image files
    for x_img in files_original:
        for y_img in files_hand:
            if (
                fnmatch.fnmatch(x_img, "*.png")
                and fnmatch.fnmatch(y_img, "*.png")
                and x_img == y_img
            ):
                rta.append(x_img)

    return rta


def openDataSet(dataset):
    Console.info("Opening dataset...")
    file_name = "img-for-autoencoder-" + dataset + ".hdf5"
    path_to_save = os.path.join(__location__, "dataset", file_name)
    with h5py.File(path_to_save, "r+") as f:
        X_img = f["x_img"][()]
        y_img = f["y_img"][()]
        f.close()

    return X_img, y_img


# Usado en caso de usar multiples core
output = multiprocessing.Queue()


def mpStart(files, output, progress_num):
    output.put(loadDataSet(files))


def progressFiles(files):
    total_file = len(files)
    Console.info("Image total:", total_file)

    num_processes = multiprocessing.cpu_count()
    if platform.system() == "Linux" and num_processes > 1:
        processes = []

        lot_size = int(total_file / num_processes)

        for x in range(1, num_processes + 1):
            if x < num_processes:
                lot_img = files[(x - 1) * lot_size : ((x - 1) * lot_size) + lot_size]
            else:
                lot_img = files[(x - 1) * lot_size :]
            processes.append(Process(target=mpStart, args=(lot_img, output, x)))

        if len(processes) > 0:
            Console.info("Get histogram of the images...")
            for p in processes:
                p.start()

            result = []
            for x in range(num_processes):
                result.append(output.get(True))

            for p in processes:
                p.join()

            X_img = []
            y_img = []
            for mp_X_img, mp_y_img in result:
                X_img = X_img + mp_X_img
                y_img = y_img + mp_y_img
            updateProgress(1, total_file, total_file, "")

            return X_img, y_img
    else:
        Console.info("No podemos dividir la cargan en distintos procesadores")
        exit(0)


if __name__ == "__main__":
    # Make two folder for hands and not_hands, with histogram values on csv file
    if MAKE_HANDS_FROM_HUMAN:
        makeHandsHuman()

    files = getFiles()
    X_img, y_img = progressFiles(files)
    saveDataSet(X_img, y_img)

    X_img, y_img = openDataSet("training")
    Console.log("Dataset", len(X_img), len(y_img))
