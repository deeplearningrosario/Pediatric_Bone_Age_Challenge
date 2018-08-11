#!/usr/bin/python3
# ./deep_cut_hands.py -lw ./model-backup/cut-hand/model.h5

from keras.applications import InceptionV3, ResNet50, Xception
from keras.layers import Flatten, Dense, Input, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad
from six.moves import cPickle
import argparse
import cv2
import fnmatch
import keras
import numpy as np
import os
import pandas as pd
import sys

# Directory of dataset to use
TRAIN_DIR = "dataset_sample"
# TRAIN_DIR = "boneage-training-dataset"

# Use N images of dataset, If it is -1 using all dataset
CUT_DATASET = 1000

# network and training
EPOCHS = 500
BATCH_SIZE = 15

# https://keras.io/optimizers
# OPTIMIZER = Adam(lr=0.001)
OPTIMIZER = RMSprop()
# OPTIMIZER = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# OPTIMIZER = Adagrad(lr=0.05)

# Sort dataset randomly
SORT_RANDOMLY = True

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-lw", "--load_weights", help="Path to the file weights")
ap.add_argument("-tb", "--tensorBoard", help="Active tensorBoard")
ap.add_argument("-rl", "--reduce_learning", help="Active reduce learning rate")
ap.add_argument("-not_cp", "--not_checkpoint", help="Active checkpoint")
ap.add_argument("-t", "--train", help="Run train model")
args = vars(ap.parse_args())

# For this problem the validation and test data provided by the concerned authority did not have labels,
# so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

train_dir = os.path.join(__location__, TRAIN_DIR)

img_file = ""

# Turn saving renders feature on/off
SAVE_RENDERS = False

# Create intermediate images in separate folders for debugger.
# mask, cut_hand, delete_object, render
SAVE_IMAGE_FOR_DEBUGGER = False


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
    print("\n[INFO] Get histogram of the images...")
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
                print("\n[INFO] Create folder", folder)
                os.makedirs(os.path.join(__location__, TRAIN_DIR, folder))
    if SAVE_RENDERS:
        if not os.path.exists(os.path.join(__location__, TRAIN_DIR, "render")):
            print("\n[INFO] Create folder render")
            os.makedirs(os.path.join(__location__, TRAIN_DIR, "render"))


def loadCallBack():
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath="weights_deep_cut_hands/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
        save_weights_only=True,
        period=1,
    )

    # Reduce learning rate
    reduceLROnPlat = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.8, patience=3, verbose=1, min_lr=0.0001
    )

    # TensorBoard
    # how to use: $ tensorboard --logdir path_to_current_dir/Graph
    # Save log for tensorboard
    LOG_DIR_TENSORBOARD = os.path.join(__location__, "tensorboard")
    if not os.path.exists(LOG_DIR_TENSORBOARD):
        os.makedirs(LOG_DIR_TENSORBOARD)

    tbCallBack = keras.callbacks.TensorBoard(
        log_dir=LOG_DIR_TENSORBOARD,
        batch_size=BATCH_SIZE,
        histogram_freq=0,
        write_graph=True,
        write_images=True,
    )
    print("tensorboard --logdir", LOG_DIR_TENSORBOARD)

    cb = []

    if args["tensorBoard"] != None:
        cb.append(tbCallBack)
    if args["not_checkpoint"] == None:
        cb.append(checkpoint)
    if args["reduce_learning"] != None:
        cb.append(reduceLROnPlat)

    return cb


def makerModel():
    # First we need to create a model structure
    hist_input = Input(shape=(256,), name="hist_input")

    x_lower = Dense(256, activation="sigmoid")(hist_input)
    x_lower = Dense(256, activation="relu")(x_lower)
    #x_lower = Dense(200, activation="relu")(x_lower)
    x_lower = Dense(128, activation="relu")(x_lower)
    #x_lower = Dense(32, activation="relu")(x_lower)
    #x_lower = Dense(16, activation="relu")(x_lower)

    x_upper = Dense(256, activation="sigmoid")(hist_input)
    x_upper = Dense(256, activation="relu")(x_upper)
    #x_upper = Dense(200, activation="relu")(x_upper)
    x_upper = Dense(128, activation="relu")(x_upper)
    #x_upper = Dense(32, activation="relu")(x_upper)
    #x_upper = Dense(2, activation="relu")(x_upper)

    # Prediction for the upper and lower value
    lower_output = Dense(1, name="lower")(x_lower)
    upper_output = Dense(1, name="upper")(x_upper)

    model = Model(inputs=[hist_input], outputs=[lower_output, upper_output])

    model.compile(loss="mean_squared_error", metrics=["MAE", "MSE"], optimizer=OPTIMIZER)

    print("\n[INFO] Model summary")
    print(model.summary())

    # Load weight
    if args["load_weights"] != None:
        print("Loading weights from", args["load_weights"])
        model.load_weights(args["load_weights"])

    return model


def trainModel(model, X_train, y_lower, y_upper):
    print("\n[INFO] Create validation sets, training set, testing set...")
    # Split images dataset
    k = int(len(X_train) / 6)  # Decides split count

    hist_test = X_train[:k]
    lower_test = y_lower[:k]
    upper_test = y_upper[:k]

    hist_valid = X_train[k: 2 * k]
    lower_valid = y_lower[k: 2 * k]
    upper_valid = y_upper[k: 2 * k]

    hist_train = X_train[2 * k:]
    lower_train = y_lower[2 * k:]
    upper_train = y_upper[2 * k:]

    print("\n[INFO] Sizes of the new set")
    print("hist_train:", len(hist_train))
    print("lower_train:", len(lower_train))
    print("upper_train:", len(upper_train))
    print("hist_valid:", len(hist_valid))
    print("lower_valid:", len(lower_valid))
    print("upper_valid:", len(upper_valid))
    print("hist_test:", len(hist_test))
    print("lower_test:", len(lower_test))
    print("upper_test:", len(upper_test))

    # Save weights after every epoch
    if not os.path.exists(os.path.join(__location__, "weights_deep_cut_hands")):
        os.makedirs(os.path.join(__location__, "weights_deep_cut_hands"))

    print("\n[INFO] Training network...")
    history = model.fit(
        [hist_train],
        [lower_train, upper_train],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        validation_data=([hist_valid], [lower_valid, upper_valid]),
        callbacks=loadCallBack()
    )

    print("\n[INFO] Save model to disck...")
    # Path to save model
    PATHE_SAVE_MODEL = os.path.join(__location__, "model-backup", "cut-hand")

    # Save weights after every epoch
    if not os.path.exists(PATHE_SAVE_MODEL):
        os.makedirs(PATHE_SAVE_MODEL)

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(os.path.join(PATHE_SAVE_MODEL, "model.yaml"), "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(os.path.join(PATHE_SAVE_MODEL, "model.h5"))
    print("OK")

    # evaluate the network
    print("\n[INFO] Evaluating network...")
    score = model.evaluate(
        [hist_test], [lower_test, upper_test], batch_size=BATCH_SIZE, verbose=1
    )

    print("Test loss:", score[1], score[4])
    print("Test MAE:", score[3], score[5])
    print("Test MSE:", score[0], score[2])

    # list all data in history
    print("\n[INFO] Save model history graphics...")
    print(history.history.keys())


# Como vamos a usar multi procesos uno por core.
# Los procesos hijos cargan el mismo código.
# Este if permite que solo se ejecute lo que sigue si es llamado
# como proceso raíz.
if __name__ == "__main__":
    checkPath()

    files = getFiles()
    (X_train, y_lower, y_upper) = loadDataSet(files)

    model = makerModel()
    if args["train"] != None:
        trainModel(model, X_train, y_lower, y_upper)
    else:
