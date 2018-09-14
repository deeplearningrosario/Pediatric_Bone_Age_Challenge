#!/usr/bin/python3
import cv2
import math
import numpy as np
import os
import pandas as pd
import h5py
import sys

# Directory of dataset to use
# TRAIN_DIR = "dataset_sample"
TRAIN_DIR = "boneage-training-dataset"

# Use N images of dataset, If it is -1 using all dataset
CUT_DATASET = -1

# Remove images that are less than or equal to 23 months of age
REMOVE_AGE = 23

# Usgin for auto-encoder
GENERATE_IMAGE_FOR_AUTOENCODER = False

# Image resize
# IMAGE_SIZE = (299, 299)
IMAGE_SIZE = (224, 224)

# Turn saving renders feature on/off
SAVE_RENDERS = False

# Create intermediate images in separate folders for debugger.
# mask, cut_hand, delete_object, render
SAVE_IMAGE_FOR_DEBUGGER = False

# Extracting hands from images and using that new dataset.
# Simple dataset is correct, I am verifying the original.
EXTRACTING_HANDS = True

# Turn rotate image on/off
ROTATE_IMAGE = True

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
def histogramsLevelFix(img, min_color, max_color):
    # This function is only prepared for images in scale of gripes
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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


# Cut the hand of the image
# Look for the largest objects and create a mask, with that new mask
# is applied to the original and cut out.
def cutHand(image):
    image_copy = image.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(image)

    img = cv2.medianBlur(img, 5)
    th2 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    th3 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    thresh = cv2.bitwise_not(th3, th2)
    thresh = cv2.GaussianBlur(thresh, (5, 5), 0)

    (_, contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # I guess the largest object is the hand or the only object in the image.
    largest_object_index = 0
    for i, cnt in enumerate(contours):
        if cv2.contourArea(contours[largest_object_index]) < cv2.contourArea(cnt):
            largest_object_index = i

    # create bounding rectangle around the contour (can skip below two lines)
    [x, y, w, h] = cv2.boundingRect(contours[largest_object_index])
    # White background below the largest object
    cv2.rectangle(image, (x, y), (x + w, y + h), (255), -1)

    cv2.drawContours(
        image,  # image,
        contours,  # objects
        largest_object_index,  # índice de objeto (-1, todos)
        (255),  # color
        -1,  # tamaño del borde (-1, pintar adentro)
    )

    # Trim that object of mask and image
    mask = image[y : y + h, x : x + w]
    image_cut = image_copy[y : y + h, x : x + w]

    # Apply mask
    image_cut = cv2.bitwise_and(image_cut, image_cut, mask=mask)

    writeImage("cut_hand", np.hstack([image_cut]))  # show the images ===========

    return image_cut


def rotateImage(imageToRotate):
    edges = cv2.Canny(imageToRotate, 50, 150, apertureSize=3)
    # Obtener una línea de la imágen
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)
    if not (lines is None) and len(lines) >= 1:
        for i in range(1):
            for rho, theta in lines[i]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                # cv2.line(imageToRotate, (x1, y1), (x2, y2), (0, 0, 255), 2)
                angle = math.atan2(y1 - y2, x1 - x2)
                angleDegree = (angle * 180) / math.pi
            if angleDegree < 0:
                angleDegree = angleDegree + 360
            if angleDegree >= 0 and angleDegree < 45:
                angleToSubtract = 0
            elif angleDegree >= 45 and angleDegree < 135:
                angleToSubtract = 90
            elif angleDegree >= 135 and angleDegree < 225:
                angleToSubtract = 180
            elif angleDegree >= 225 and angleDegree < 315:
                angleToSubtract = 270
            else:
                angleToSubtract = 0
            angleToRotate = angleDegree - angleToSubtract
            num_rows, num_cols = imageToRotate.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D(
                (num_cols / 2, num_rows / 2), angleToRotate, 1
            )
            imageToRotate = cv2.warpAffine(
                imageToRotate, rotation_matrix, (num_cols, num_rows)
            )
    return imageToRotate


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
    if progress == 1:
        print("")


def getColorsHands(img):
    # Find the acceptable limits of the intensity histogram
    min_color, max_color = np.percentile(img, (2.5, 99.4))
    min_color = int(min_color)
    max_color = int(max_color)

    return (min_color, max_color)


def processImage(img_path):
    # Read a image
    img = cv2.imread(img_path, 0)

    # Adjust color levels
    min_color, max_color = getColorsHands(img)
    img = histogramsLevelFix(img, min_color, max_color)

    if EXTRACTING_HANDS:
        # Trim the hand of the image
        img = cutHand(img)

    if ROTATE_IMAGE:
        # Rotate hands
        img = rotateImage(img)

    # ====================== show the images ================================
    if SAVE_IMAGE_FOR_DEBUGGER or SAVE_RENDERS:
        cv2.imwrite(os.path.join(__location__, TRAIN_DIR, "render", img_file), img)

    # Resize the images
    img = cv2.resize(img, IMAGE_SIZE)
    # Return to original colors
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Convert the image into an 8 bit array
    return np.asarray(img, dtype=np.float32)


def loadDataSet(files=[]):
    global img_file
    X_train = []
    x_gender = []
    y_age = []

    total_file = len(files)
    for i in range(total_file):
        (img_file, bone_age, gender) = files[i]
        img_file = str(img_file) + ".png"
        # Update the progress bar
        progress = float(i / total_file), (i + 1)
        updateProgress(progress[0], progress[1], total_file, img_file)

        # Get image's path
        img_path = os.path.join(train_dir, img_file)
        if os.path.exists(img_path):
            img = processImage(img_path) / 255.
            X_train.append(img)
            x_gender.append(gender)
            y_age.append(bone_age)

    updateProgress(1, total_file, total_file, img_file)

    return X_train, x_gender, y_age


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


# Save dataset
def saveDataSet(genderType, X_train, x_gender, y_age):
    print("Divide the data set...")
    img = np.asarray(X_train)
    gender = np.asarray(x_gender, dtype=np.uint8)
    age = np.asarray(y_age, dtype=np.uint8)
    # Split images dataset
    k = int(len(X_train) / 6)
    if GENERATE_IMAGE_FOR_AUTOENCODER:
        writeFile(genderType, "testing", img[:k, :, :], gender[:k], age[:k])
        writeFile(
            genderType,
            "validation",
            img[k : 2 * k, :, :],
            gender[k : 2 * k],
            age[k : 2 * k],
        )
        writeFile(
            genderType, "training", img[2 * k :, :, :], gender[2 * k :], age[2 * k :]
        )
    else:
        writeFile(genderType, "testing", img[:k, :, :, :], gender[:k], age[:k])
        writeFile(
            genderType,
            "validation",
            img[k : 2 * k, :, :, :],
            gender[k : 2 * k],
            age[k : 2 * k],
        )
        writeFile(
            genderType, "training", img[2 * k :, :, :, :], gender[2 * k :], age[2 * k :]
        )


# list all the image files and randomly unravel them,
# in each case you take the first N from the unordered list
def getFiles():
    print("Read csv on", TRAIN_DIR)
    female = []
    male = []
    # Read csv
    df = pd.read_csv(os.path.join(train_dir, "boneage-training-dataset.csv"))

    for index, row in df.iterrows():
        # Cut list of file
        if CUT_DATASET <= 0 or (len(female) + len(male)) < CUT_DATASET:
            # Get bone age
            bone_age = row.boneage
            if bone_age > REMOVE_AGE:
                # Get gender
                if row.male:
                    male.append((row.id, bone_age, 1))
                else:
                    female.append((row.id, bone_age, 0))

    return female, male


# Create the directories to save the images
def checkPath():
    if not os.path.exists(os.path.join(__location__, "packaging-dataset")):
        os.makedirs(os.path.join(__location__, "packaging-dataset"))
    if not os.path.exists(
        os.path.join(__location__, "packaging-dataset", "for_autoencoder")
    ):
        os.makedirs(os.path.join(__location__, "packaging-dataset", "for_autoencoder"))

    if SAVE_IMAGE_FOR_DEBUGGER:
        for folder in ["histograms_level_fix", "cut_hand", "mask"]:
            if not os.path.exists(os.path.join(__location__, TRAIN_DIR, folder)):
                os.makedirs(os.path.join(__location__, TRAIN_DIR, folder))
    if SAVE_RENDERS or SAVE_IMAGE_FOR_DEBUGGER:
        if not os.path.exists(os.path.join(__location__, TRAIN_DIR, "render")):
            os.makedirs(os.path.join(__location__, TRAIN_DIR, "render"))


# Como vamos a usar multi procesos uno por core.
# Los procesos hijos cargan el mismo código.
# Este if permite que solo se ejecute lo que sigue si es llamado
# como proceso raíz.
if __name__ == "__main__":
    checkPath()

    female, male = getFiles()

    print("Processing female images...")
    (X_train, x_gender, y_age) = loadDataSet(female)
    saveDataSet("female", X_train, x_gender, y_age)

    print("Processing male images...")
    (X_train, x_gender, y_age) = loadDataSet(male)
    saveDataSet("male", X_train, x_gender, y_age)
