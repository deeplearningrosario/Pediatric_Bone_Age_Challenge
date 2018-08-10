#!/usr/bin/python3

import cv2
import fnmatch
import numpy as np
import os
import pandas as pd
import sys

# Directory of dataset to use
TRAIN_DIR = "dataset_sample"
# TRAIN_DIR = "boneage-training-dataset"

# Use N images of dataset, If it is -1 using all dataset
CUT_DATASET = 1000
# Sort dataset randomly
SORT_RANDOMLY = not True

# Turn saving renders feature on/off
SAVE_RENDERS = False

# Create intermediate images in separate folders for debugger.
# mask, cut_hand, delete_object, render
SAVE_IMAGE_FOR_DEBUGGER = False

# Extracting hands from images and using that new dataset.
# Simple dataset is correct, I am verifying the original.
EXTRACTING_HANDS = True

# For this problem the validation and test data provided by the concerned authority did not have labels,
# so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

train_dir = os.path.join(__location__, TRAIN_DIR)

img_file = ""


# Show the images
def writeImage(path, image, force=False):
    if SAVE_IMAGE_FOR_DEBUGGER or force:
        cv2.imwrite(os.path.join(__location__, TRAIN_DIR, path, img_file), image)


# Auto adjust levels colors
# We order the colors of the image with their frequency and
# obtain the accumulated one, then we obtain the colors that
# accumulate 2.5% and 99.4% of the frequency.
def histogramsLevelFix(img):
    # This function is only prepared for images in scale of gripes
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the acceptable limits of the intensity histogram
    min_color, max_color = np.percentile(img, (2.5, 99.4))
    min_color = int(min_color)
    max_color = int(max_color)

    # To improve the preform we created a color palette with the new values
    colors_palette = []
    # Auxiliary calculation, avoid doing calculations within the 'for'
    dif_color = 255 / (max_color - min_color)
    for color in range(256):
        if color <= min_color:
            colors_palette.append(0)
        elif color >= max_color:
            colors_palette.append(255)
        else:
            colors_palette.append(int(round((color - min_color) * dif_color)))

    # We paint the image with the new color palette
    height, width = img.shape
    for y in range(0, height):
        for x in range(0, width):
            color = img[y, x]
            img[y, x] = colors_palette[color]

    writeImage("histograms_level_fix", np.hstack([img]))  # show the images ===========

    return img


# Show a progress bar
def updateProgress(progress, tick="", total="", status="Loading..."):
    lineLength = 80
    barLength = 23
    if isinstance(progress, int):
        progress = float(progress)
    if progress < 0:
        progress = 0
        status = "Waiting...\r"
    if progress >= 1:
        progress = 1
        status = "Completed loading data\r\n"
    block = int(round(barLength * progress))
    line = str("\rImage: {0}/{1} [{2}] {3}% {4}").format(
        tick,
        total,
        str(("#" * block)) + str("." * (barLength - block)),
        round(progress * 100, 1),
        status,
    )
    emptyBlock = lineLength - len(line)
    emptyBlock = " " * emptyBlock if emptyBlock > 0 else ""
    sys.stdout.write(line + emptyBlock)
    sys.stdout.flush()


def processImage(img_path):
    # Read a image
    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist, _ = np.histogram(img, 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    return cdf_normalized


def loadDataSet(files=[]):
    global img_file
    X_train = []
    y_lower = []
    y_upper = []

    total_file = len(files)
    for i in range(total_file):
        (img_file, lower, upper) = files[i]
        # Update the progress bar
        progress = float(i / total_file), (i + 1)
        updateProgress(progress[0], progress[1], total_file, img_file)

        # Get image's path
        img_path = os.path.join(train_dir, img_file)

        X_train.append(processImage(img_path))
        y_lower.append(lower)
        y_upper.append(upper)

    updateProgress(1, total_file, total_file, img_file)

    return X_train, y_lower, y_upper


# list all the image files and randomly unravel them,
# in each case you take the first N from the unordered list
def getFiles():
    rta = []
    # Read csv
    df = pd.read_csv(os.path.join(train_dir, "histograma-dataset.csv"))

    # file names on train_dir
    files = os.listdir(train_dir)
    # filter image files
    files = [f for f in files if fnmatch.fnmatch(f, "*.png")]
    # Sort randomly
    if SORT_RANDOMLY:
        np.random.shuffle(files)

    for file_name in files:
        # Cut list of file
        if CUT_DATASET <= 0 or len(rta) < CUT_DATASET:
            # Get row with id equil image's name
            csv_row = df[df.id == int(file_name[:-4])]
            # Get lower color
            lower = csv_row.lower.tolist()[0]
            # Get upper color
            upper = csv_row.upper.tolist()[0]

            rta.append((file_name, lower, upper))

    return rta


# Create the directories to save the images
def checkPath():
    if SAVE_IMAGE_FOR_DEBUGGER:
        for folder in ["histograms_level_fix", "cut_hand", "render", "mask"]:
            if not os.path.exists(os.path.join(__location__, TRAIN_DIR, folder)):
                os.makedirs(os.path.join(__location__, TRAIN_DIR, folder))
    if SAVE_RENDERS:
        if not os.path.exists(os.path.join(__location__, TRAIN_DIR, "render")):
            os.makedirs(os.path.join(__location__, TRAIN_DIR, "render"))


# Como vamos a usar multi procesos uno por core.
# Los procesos hijos cargan el mismo código.
# Este if permite que solo se ejecute lo que sigue si es llamado
# como proceso raíz.
if __name__ == "__main__":
    checkPath()

    files = getFiles()
    (X_train, y_lower, y_upper) = loadDataSet(files)

    print()
