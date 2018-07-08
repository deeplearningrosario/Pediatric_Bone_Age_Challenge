#!/usr/bin/python3

from six.moves import cPickle
import cv2
import fnmatch
import re
import numpy as np
import os
import pandas as pd
import sys


# whitepatch, normaliza la los colores de la imagen
#
# Como las imagenes tiene diferentes tonalidades de colores
# Este algoritmo whit-patch pretende llevar los colores de la
# imagenes a un tono igual
def whitePatch(image):
    B, G, R = cv2.split(image)

    red = cv2.equalizeHist(R)
    green = cv2.equalizeHist(G)
    blue = cv2.equalizeHist(B)

    imgOut = cv2.merge((blue, green, red))

    # show the images
    # cv2.imwrite("white-patch.png", np.hstack([image, imgOut, ]))

    return imgOut


# Show a progress bar
def updateProgress(progress, tick='', total='', status='Loading...'):
    barLength = 45
    if isinstance(progress, int):
        progress = float(progress)
    if progress < 0:
        progress = 0
        status = "Waiting...\r\n"
    if progress >= 1:
        progress = 1
        status = "Completed loading data\r\n"
    block = int(round(barLength * progress))
    sys.stdout.write(str("\rImage: {0}/{1} [{2}] {3}% {4}").format(
        tick,
        total,
        str(("#" * block)) + str("." * (barLength - block)),
        round(progress * 100, 1), status))
    sys.stdout.flush()


# For this problem the validation and test data provided by the concerned authority did not have labels, so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
train_dir = os.path.join(__location__, 'dataset_sample')

X_train = []
y_age = []
y_gender = []

df = pd.read_csv(os.path.join(train_dir, 'dataset_sample_labels.csv'))
a = df.values
m = a.shape[0]

print('Loading data set...')
# file names on train_dir
files = os.listdir(train_dir)
# filtter image file
files = [f for f in files if fnmatch.fnmatch(f, '*.png')]
totalFile = len(files)

for i in range(totalFile):
    imgFile = files[i]

    y_age.append(df.boneage[df.id == int(imgFile[:-4])].tolist()[0])
    a = df.male[df.id == int(imgFile[:-4])].tolist()[0]
    if a:
        y_gender.append(1)
    else:
        y_gender.append(0)

    img_path = os.path.join(train_dir, imgFile)
    img = cv2.imread(img_path)

    imgBGR2RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgColorEqualize = whitePatch(imgBGR2RGB)

    # =========================================================
    lower = np.array([128, 128, 128])
    upper = np.array([255, 255, 255])
    # find the colors within the specified boundaries and apply the mask
    # mask = cv2.imread('./mask.png')
    # mask = cv2.inRange(imgColorEqualize, lower, upper)
    # ========================
    # imgBlackWihte = cv.Threshold(imgColorEqualize, img, threshold, 255, cv.CV_THRESH_BINARY | cv.CV_THRESH_OTSU);
    mask = cv2.GaussianBlur(imgColorEqualize, (23, 23), 0)
    mask = cv2.inRange(mask, lower, upper)
    # mask = cv2.dilate(mask, None, iterations=23)
    # =========================

    output = cv2.bitwise_and(imgBGR2RGB, imgBGR2RGB, mask=mask)

    # show the images
    cv2.imwrite(
        "remainder.png",
        np.hstack([
            img,
            imgBGR2RGB,
            imgColorEqualize,
            output
        ])
    )
    # img = cv2.resize(img, (224, 224))

    x = np.asarray(img, dtype=np.uint8)
    X_train.append(x)

    # Update the progress bar
    updateProgress(float((i+1) / totalFile), (i+1), totalFile, imgFile)


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
