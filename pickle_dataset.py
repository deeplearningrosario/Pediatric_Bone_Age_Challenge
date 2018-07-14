#!/usr/bin/python3

from six.moves import cPickle
import cv2
import fnmatch
import numpy as np
import os
import pandas as pd
import sys

# Turn saving renders feature on/off
SAVE_RENDERS = False

# Create intermediate images in separate folders for debugger.
# mask, cut_hand, delete_object, render
SAVE_IMAGE_FOR_DEBUGGER = False

# Extracting hands from images and using that new dataset.
# Simple dataset is correct, I am verifying the original.
EXTRACTING_HANDS = False


# Delete small objects from the images
def deleteObjects(image):
    # Detect the edges with Canny
    img = cv2.Canny(image, 100, 400)  # 50,150  ; 100,500

    # We look for contours
    (_, contours, _) = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # In case of having more than 10000 contours, sub figures
    if len(contours) > 10000:
        # Create a kernel of '1' of 10x10, used as an eraser
        kernel = np.ones((10, 10), np.uint8)
        # Transformation is applied to eliminate particles
        img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Get a new mask with fewer objects
        _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_OTSU)

        # Detect the edges with Canny and then we look for contours
        img = cv2.Canny(thresh, 100, 400)  # 50,150  ; 100,500
        (_, contours, _) = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        image = cv2.bitwise_and(image, image, mask=thresh)

        # He reported it because it can take a long time if the number is large
        if len(contours) > 3000:
            # print(' Img with', len(contours), 'contours')
            updateProgress(progress[0], progress[1], total_file,
                           img_file + " Img with " + str(len(contours)) + " contours")
        # ================================================================
        if SAVE_IMAGE_FOR_DEBUGGER:
            # show the images
            cv2.imwrite(
                os.path.join(__location__, "dataset_sample", "delete_object", img_file),
                np.hstack([
                    thresh,
                    img,
                    # image
                ])
            )
        # ================================================================

    if len(contours) > 1:
        # I guess the largest object is the hand or the only object in the image
        # From the contour list search the index of the largest object
        largest_object_index = 0
        for i, cnt in enumerate(contours):
            if cv2.contourArea(contours[largest_object_index]) < cv2.contourArea(cnt):
                largest_object_index = i

        # Paint the objects smaller than 30% of the large, limit: (0.2, 0.5]
        lenOfObjetoGrande = cv2.contourArea(contours[largest_object_index]) * 0.3
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < lenOfObjetoGrande:
                cv2.drawContours(image, contours, i, (0, 0, 0), -1)

        # Paint the largest object in white
        cv2.drawContours(
            image,  # image,
            contours,  # objects
            largest_object_index,  # índice de objeto (-1, todos)
            (255, 255, 255),  # color
            -1  # tamaño del borde (-1, pintar adentro)
        )
        # Add a border to the largest object in white
        cv2.drawContours(
            image,  # image,
            contours,  # objects
            largest_object_index,  # índice de objeto (-1, todos)
            (255, 255, 255),  # color
            10  # tamaño del borde (-1, pintar adentro)
        )

    return image, len(contours)


# Cut the hand of the image
# Look for the largest objects and create a mask, with that new mask
# is applied to the original and cut out.
def cutHand(image, original_image):
    image_copy = image.copy()

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur
    blurred = cv2.GaussianBlur(gray, (47, 47), 0)

    # thresholdin: Otsu's Binarization method
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
    thresh = cv2.GaussianBlur(thresh, (41, 41), 0)

    # ================================================================
    if SAVE_IMAGE_FOR_DEBUGGER:
        # show the images
        cv2.imwrite(
            os.path.join(__location__, "dataset_sample", "cut_hand", img_file),
            np.hstack([
                thresh,
                # mask,
                # image
            ])
        )
    # ================================================================

    (image, contours, _) = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # I guess the largest object is the hand or the only object in the image.
    largest_object_index = 0
    for i, cnt in enumerate(contours):
        if cv2.contourArea(contours[largest_object_index]) < cv2.contourArea(cnt):
            largest_object_index = i

    # create bounding rectangle around the contour (can skip below two lines)
    [x, y, w, h] = cv2.boundingRect(contours[largest_object_index])
    # Black background below the largest object
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), -1)

    cv2.drawContours(
        image,  # image,
        contours,  # objects
        largest_object_index,  # índice de objeto (-1, todos)
        (255, 255, 255),  # color
        -1  # tamaño del borde (-1, pintar adentro)
    )

    # Trim that object
    mask = image[y:y+h, x:x+w]
    image_cut = image_copy[y:y+h, x:x+w]

    output = cv2.bitwise_and(image_cut, image_cut, mask=mask)

    # Ver bien este caso, creo que no sucede
    # In case the image is black I return the original
    # gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    # if (cv2.mean(gray)[0] <= 10.0):
    #     print("\n-------------IMAGEN NEGRA----------------\n")
    #    return image_copy
    # else:
    return output


# Create a mask for the hand.
# I guess the biggest object is the hand
def createMask(image):
    # Apply a technique to normalize the overall colors of the image
    equalized_image = equalizeImg(image)
    gray = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY)

    # Blur the image to avoid erasing borders
    mask = cv2.GaussianBlur(gray, (33, 33), 0)
    mask = cv2.inRange(
        mask,
        np.array(128),  # lower color
        np.array(255)  # upper color
    )
    # We apply the mask
    image = cv2.bitwise_and(equalized_image, equalized_image, mask=mask)

    # Delete other figures
    image, contours = deleteObjects(image)

    if contours > 0:
        # Blur the image to avoid erasing borders
        image = cv2.GaussianBlur(image, (25, 25), 0)
        image = cv2.inRange(
            image,
            np.array([60, 60, 60]),  # lower color
            np.array([255, 255, 255])  # upper color
        )

        # show the images ====================================================
        if SAVE_IMAGE_FOR_DEBUGGER:
            cv2.imwrite(
                os.path.join(__location__, "dataset_sample", "mask", img_file),
                np.hstack([
                    # mask,
                    image,
                ])
            )
        # ====================================================================
    else:
        # In case of not finding an object, use the image
        image = equalized_image

    return image


# white-patch, normalizes the colors of the image
#
# How the images have different shades of colors
# This whit-patch algorithm is intended to carry the colors of the
# images to an equal tone.
def equalizeImg(image):
    B, G, R = cv2.split(image)

    red = cv2.equalizeHist(R)
    green = cv2.equalizeHist(G)
    blue = cv2.equalizeHist(B)

    imgOut = cv2.merge((blue, green, red))

    return imgOut


# Show a progress bar
def updateProgress(progress, tick='', total='', status='Loading...'):
    lineLength = 80
    barLength = 30
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
        status
    )
    emptyBlock = lineLength - len(line)
    emptyBlock = " "*emptyBlock if emptyBlock > 0 else ""
    sys.stdout.write(line + emptyBlock)
    sys.stdout.flush()


# For this problem the validation and test data provided by the concerned authority did not have labels, so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
train_dir = os.path.join(__location__, 'dataset_sample')

X_train = []
y_age = []
y_gender = []

df = pd.read_csv(os.path.join(train_dir, 'boneage-training-dataset.csv'))
a = df.values
m = a.shape[0]


# Create the directories to save the images
if SAVE_IMAGE_FOR_DEBUGGER:
    for folder in ['mask', 'cut_hand', 'delete_object', 'render']:
        if not os.path.exists(os.path.join(__location__, "dataset_sample", folder)):
            os.makedirs(os.path.join(__location__, "dataset_sample", folder))
if SAVE_RENDERS:
    if not os.path.exists(os.path.join(__location__, "dataset_sample", "render")):
        os.makedirs(os.path.join(__location__, "dataset_sample", "render"))

print('Loading data set...')
# file names on train_dir
files = os.listdir(train_dir)
# filter image files
files = [f for f in files if fnmatch.fnmatch(f, '*.png')]
total_file = len(files)

for i in range(total_file):
    img_file = files[i]

    # Update the progress bar
    progress = float(i / total_file), (i + 1)
    updateProgress(progress[0], progress[1], total_file, img_file)

    y_age.append(df.boneage[df.id == int(img_file[:-4])].tolist()[0])
    a = df.male[df.id == int(img_file[:-4])].tolist()[0]
    if a:
        y_gender.append(1)
    else:
        y_gender.append(0)

    # Read a image
    img_path = os.path.join(train_dir, img_file)
    img = cv2.imread(img_path)

    # Sort colors in R, G, B
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if EXTRACTING_HANDS:
        # Create mask to highlight your hand
        mask = createMask(img.copy())
        img_hand = cv2.bitwise_and(img, img, mask=mask)

        # Trim the hand of the image
        img = cutHand(equalizeImg(img_hand), img)

        # ====================== show the images ================================
        if SAVE_IMAGE_FOR_DEBUGGER or SAVE_RENDERS:
            cv2.imwrite(
                os.path.join(__location__, "dataset_sample", "render", img_file),
                np.hstack([
                    img
                ])
            )
        # =======================================================================

    # Resize the images
    img = cv2.resize(img, (224, 224))

    x = np.asarray(img, dtype=np.uint8)
    X_train.append(x)

updateProgress(1, total_file, total_file, img_file)

print('\nSaving data...')
# Save data
train_pkl = open('data.pkl', 'wb')
cPickle.dump(X_train, train_pkl, protocol=cPickle.HIGHEST_PROTOCOL)
train_pkl.close()

train_age_pkl = open('data_age.pkl', 'wb')
cPickle.dump(y_age, train_age_pkl, protocol=cPickle.HIGHEST_PROTOCOL)
train_age_pkl.close()

train_gender_pkl = open('data_gender.pkl', 'wb')
cPickle.dump(y_gender, train_gender_pkl, protocol=cPickle.HIGHEST_PROTOCOL)
train_gender_pkl.close()
print('\nCompleted saved data')
