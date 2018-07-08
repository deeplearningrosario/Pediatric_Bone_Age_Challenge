#!/usr/bin/python3

from six.moves import cPickle
import cv2
import fnmatch
import numpy as np
import os
import pandas as pd
import sys


# Show a progress bar
def updateProgress(progress, tick='', total='', status='loading...'):
    barLength = 45
    if isinstance(progress, int):
        progress = float(progress)
    if progress < 0:
        progress = 0
        status = "waiting...\r\n"
    if progress >= 1:
        progress = 1
        status = "completed loading data\r\n"
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
# file names
files = os.listdir(train_dir)
totalFile = len(files)

for i in range(totalFile):
    imgFile = files[i]

    if fnmatch.fnmatch(imgFile, '*.png'):
        y_age.append(df.boneage[df.id == int(imgFile[:-4])].tolist()[0])
        a = df.male[df.id == int(imgFile[:-4])].tolist()[0]
        if a:
            y_gender.append(1)
        else:
            y_gender.append(0)

        img_path = os.path.join(train_dir, imgFile)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

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
print('\n100% completed saved data')
