#!/usr/bin/python3
# ./main_histogram.py -lw ./model/model_histogram.h5
# ./main_histogram.py -lw ./model/model_histogram.h5 --train False
# ./main_histogram.py -lw ./model/model_histogram.h5 --train False --evaluate True --predict True

from keras.layers import Flatten, Dense, Input, Dropout, BatchNormalization, concatenate
from keras.models import Model
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad
from utilities import Console
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
EPOCHS = 1500
BATCH_SIZE = 17

# https://keras.io/optimizers
OPTIMIZER = Adam(lr=0.001)
# OPTIMIZER = RMSprop()
# OPTIMIZER = Adadelta(lr=0.01, rho=0.95, epsilon=None, decay=0.0)
# OPTIMIZER = Adagrad(lr=0.05)

# Show how it closed when the difference is N color
E_TOLERANCE = 20

# Sort dataset randomly
SORT_RANDOMLY = True

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-lw", "--load_weights", help="Path to the file weights")
ap.add_argument("-tb", "--tensorBoard", default="False", help="Active tensorBoard")
ap.add_argument("-cp", "--checkpoint", default="False", help="Active checkpoint")
ap.add_argument(
    "-rl", "--reduce_learning", default="False", help="Active reduce learning rate"
)

ap.add_argument("-t", "--train", default="True", help="Run train model")
ap.add_argument("-e", "--evaluate", default="False", help="Evaluating model")
ap.add_argument("-p", "--predict", help="File to predict values")
args = vars(ap.parse_args())

# For this problem the validation and test data provided by the concerned authority did not have labels,
# so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

train_dir = os.path.join(__location__, "..", TRAIN_DIR)

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


def getHistogram(img):
    hist, _ = np.histogram(img, 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    return cdf_normalized


def loadDataSet(files=[]):
    Console.info("Get histogram of the images...")
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
        # Read a image
        img = cv2.imread(img_path, 0)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        X_train.append(getHistogram(img))
        y_lower.append(lower)
        y_upper.append(upper)

    updateProgress(1, total_file, total_file, img_file)

    return X_train, y_lower, y_upper


# list all the image files and randomly unravel them,
# in each case you take the first N from the unordered list
def getFiles():
    rta = []
    # Read csv
    df = pd.read_csv(os.path.join(train_dir, "histogram-dataset.csv"))

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
            # print(file_name)
            # Get row with id equil image's name
            csv_row = df[df.id == int(file_name[:-4])]
            # Get lower color
            lower = csv_row.lower.tolist()
            # Get upper color
            upper = csv_row.upper.tolist()
            if (lower and upper) and (
                not (lower[0] != lower[0] and upper[0] != upper[0])
            ):
                rta.append((file_name, lower[0], upper[0]))
            else:
                Console.log("Not data for", file_name)

    return rta


# Create the directories to save the images
# def checkPath():
#    if SAVE_IMAGE_FOR_DEBUGGER:
#        for folder in ["histograms_level_fix", "cut_hand", "render", "mask"]:
#            if not os.path.exists(os.path.join(__location__, TRAIN_DIR, folder)):
#                Console.info("Create folder", folder)
#                os.makedirs(os.path.join(__location__, TRAIN_DIR, folder))
#    if SAVE_RENDERS:
#        if not os.path.exists(os.path.join(__location__, TRAIN_DIR, "render")):
#            Console.info("Create folder render")
#            os.makedirs(os.path.join(__location__, TRAIN_DIR, "render"))


# Make Keras callback
def loadCallBack():
    cb = []

    # TensorBoard
    # how to use: $ tensorboard --logdir path_to_current_dir/Graph
    if args["tensorBoard"] == "True":
        # Save log for tensorboard
        LOG_DIR_TENSORBOARD = os.path.join(__location__, "..", "tensorboard")
        if not os.path.exists(LOG_DIR_TENSORBOARD):
            os.makedirs(LOG_DIR_TENSORBOARD)

        tbCallBack = keras.callbacks.TensorBoard(
            log_dir=LOG_DIR_TENSORBOARD,
            batch_size=BATCH_SIZE,
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        )

        Console.info("tensorboard --logdir", LOG_DIR_TENSORBOARD)
        cb.append(tbCallBack)

    if args["checkpoint"] == "True":
        # Save weights after every epoch
        if not os.path.exists(os.path.join(__location__, "weights")):
            os.makedirs(os.path.join(__location__, "weights"))
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath="weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
            save_weights_only=True,
            period=1,
        )
        Console.info("Save weights after every epoch")
        cb.append(checkpoint)

    # Reduce learning rate
    if args["reduce_learning"] == "True":
        reduceLROnPlat = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.8, patience=3, verbose=1, min_lr=0.0001
        )
        Console.info("Add Reduce learning rate")
        cb.append(reduceLROnPlat)

    return cb


def makerModel():
    # First we need to create a model structure
    hist_input = Input(shape=(256,), name="hist_input")

    x1_lower = Dense(256, activation="sigmoid")(hist_input)
    x1_lower = Dense(256, activation="relu")(x1_lower)
    x1_lower = Dense(64, activation="relu")(x1_lower)

    x2_lower = Dense(256, activation="sigmoid")(hist_input)
    x2_lower = Dense(256, activation="relu")(x2_lower)
    x2_lower = Dense(64, activation="relu")(x2_lower)

    x1_upper = Dense(256, activation="sigmoid")(hist_input)
    x1_upper = Dense(256, activation="relu")(x1_upper)
    x1_upper = Dense(64, activation="relu")(x1_upper)

    x2_upper = Dense(256, activation="sigmoid")(hist_input)
    x2_upper = Dense(256, activation="relu")(x2_upper)
    x2_upper = Dense(64, activation="relu")(x2_upper)

    x_lower = concatenate([x1_lower, x2_lower])
    x_lower = Dense(128, activation="relu")(x_lower)

    x_upper = concatenate([x1_upper, x2_upper])
    x_upper = Dense(128, activation="relu")(x_upper)

    # Prediction for the upper and lower value
    lower_output = Dense(1, name="lower")(x_lower)
    upper_output = Dense(1, name="upper")(x_upper)

    model = Model(inputs=[hist_input], outputs=[lower_output, upper_output])

    model.compile(loss="mean_squared_error", metrics=["MAE", "MSE"], optimizer=OPTIMIZER)

    if __name__ == "__main__":
        Console.info("Model summary")
        print(model.summary())

    # Load weight
    if args["load_weights"] != None:
        Console.info("Loading weights from", args["load_weights"])
        model.load_weights(args["load_weights"])

    return model


def trainModel(model, X_train, y_lower, y_upper):
    Console.info("Create validation sets, training set, testing set...")
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

    Console.info("Sizes of the new set")
    print("hist_train:", len(hist_train))
    print("lower_train:", len(lower_train))
    print("upper_train:", len(upper_train))
    print("hist_valid:", len(hist_valid))
    print("lower_valid:", len(lower_valid))
    print("upper_valid:", len(upper_valid))
    print("hist_test:", len(hist_test))
    print("lower_test:", len(lower_test))
    print("upper_test:", len(upper_test))

    Console.info("Training network...")
    history = model.fit(
        [hist_train],
        [lower_train, upper_train],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        validation_data=([hist_valid], [lower_valid, upper_valid]),
        callbacks=loadCallBack(),
    )

    Console.info("Save model to disck...")
    # Path to save model
    PATHE_SAVE_MODEL = os.path.join(__location__, "model")

    # Save weights after every epoch
    if not os.path.exists(PATHE_SAVE_MODEL):
        os.makedirs(PATHE_SAVE_MODEL)

    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(os.path.join(PATHE_SAVE_MODEL, "model_histogram.yaml"), "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(os.path.join(PATHE_SAVE_MODEL, "model_histogram.h5"))
    print("OK")

    # evaluate the network
    Console.info("Evaluating network...")
    score = model.evaluate(
        [hist_test], [lower_test, upper_test], batch_size=BATCH_SIZE, verbose=1
    )

    Console.log("Test loss:", score[1], score[4])
    Console.log("Test MAE:", score[3], score[5])
    Console.log("Test MSE:", score[0], score[2])

    # list all data in history
    Console.info("Save model history graphics...")
    print(history.history.keys())


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
        print()


# Como vamos a usar multi procesos uno por core.
# Los procesos hijos cargan el mismo código.
# Este if permite que solo se ejecute lo que sigue si es llamado
# como proceso raíz.
if __name__ == "__main__":
    # Check if exixt folder
    # checkPath()

    if args["predict"] == None or args["predict"] == "True":
        files = getFiles()
        (X_train, y_lower, y_upper) = loadDataSet(files)

    # Create model
    model = makerModel()

    if args["train"] == "True":
        trainModel(model, X_train, y_lower, y_upper)

    if args["evaluate"] == "True":
        Console.info("Evaluating model...")
        score = model.evaluate(
            [X_train], [y_lower, y_upper], batch_size=BATCH_SIZE, verbose=1
        )
        Console.log("Test loss:", score[1], score[4])
        Console.log("Test MAE:", score[3], score[5])
        Console.log("Test MSE:", score[0], score[2])

    if args["predict"] != None and args["predict"] != "False":
        if args["predict"] != "True":
            Console.info("Predict for", args["predict"])
        else:
            Console.info("Predict...")
            # new instance where we do not know the answer
            Xnew = X_train
            Xnew = np.array(Xnew)

            # make a prediction
            ynew = model.predict(Xnew)

            for i in range(len(files)):
                (name, x_lower, x_upper) = files[i]
                lower = int(ynew[0][i])
                upper = int(ynew[1][i])
                e_lower = x_lower - lower
                e_upper = x_upper - upper

                e_lower = e_lower > E_TOLERANCE or e_lower < -E_TOLERANCE
                e_upper = e_upper > E_TOLERANCE or e_upper < -E_TOLERANCE

                if e_lower or e_upper:
                    lower = Console.Red + str(lower) + Console.NC if e_lower else lower
                    upper = Console.Red + str(upper) + Console.NC if e_upper else upper
                    # show the inputs and predicted outputs
                    Console.error(
                        "File %s, Lower: %s, Upper: %s" % (files[i], lower, upper)
                    )
                else:
                    Console.log("File %s, Lower: %s, Upper: %s" % (name, lower, upper))
