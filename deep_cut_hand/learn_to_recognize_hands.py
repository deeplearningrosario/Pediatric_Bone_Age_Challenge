#!/usr/bin/python3
# ./learn_to_recognize_hands.py -lw ./model/model_histogram.h5
# ./learn_to_recognize_hands.py -lw ./model/model_histogram.h5 --train False
# ./learn_to_recognize_hands.py -lw ./model/model_histogram.h5 --train False --evaluate True --predict True

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, SGD
from utilities import Console, updateProgress
from get_hands_dataset import openDataSet
import math
import argparse
import keras
import numpy as np
import os

# network and training
EPOCHS = 150
BATCH_SIZE = 7

# https://keras.io/optimizers
# OPT = Adam(lr=0.001)
# OPT = RMSprop()
OPT = SGD(lr=0.01, clipvalue=0.5)
# OPT = Adadelta(lr=0.01, rho=0.95, epsilon=None, decay=0.0)
# OPT = Adagrad(lr=0.05)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-lw", "--load_weights", help="Path to the file weights")
ap.add_argument("-tb", "--tensorBoard", default="False", help="Active tensorBoard")
ap.add_argument("-cp", "--checkpoint", default="False", help="Active checkpoint")
ap.add_argument(
    "-rl", "--reduce_learning", default="True", help="Active reduce learning rate"
)

ap.add_argument("-t", "--train", default="True", help="Run train model")
ap.add_argument("-e", "--evaluate", default="False", help="Evaluating model")
ap.add_argument("-p", "--predict", help="File to predict values")
args = vars(ap.parse_args())

# For this problem the validation and test data provided by the concerned authority did not have labels,
# so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# Turn saving renders feature on/off
SAVE_RENDERS = False

# Create intermediate images in separate folders for debugger.
# mask, cut_hand, delete_object, render
SAVE_IMAGE_FOR_DEBUGGER = False


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
    model = Sequential()
    # First we need to create a model structure
    model.add(Dense(256, input_dim=256, name="hist", activation="sigmoid"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(1, name="hands", activation="sigmoid"))

    # Compile model
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=OPT)

    if __name__ == "__main__":
        Console.info("Model summary")
        print(model.summary())

    # Load weight
    if args["load_weights"] != None:
        Console.info("Loading weights from", args["load_weights"])
        model.load_weights(args["load_weights"])

    return model


def trainModel(model, X_train, y_train):
    Console.info("Create validation sets, training set, testing set...")
    # Split images dataset
    k = int(len(X_train) / 6)  # Decides split count

    hist_test = X_train[:k]
    hand_test = y_train[:k]

    hist_valid = X_train[k: 2 * k]
    hand_valid = y_train[k: 2 * k]

    hist_train = X_train[2 * k:]
    hand_train = y_train[2 * k:]

    Console.info("Training network...")
    history = model.fit(
        hist_train,
        hand_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        validation_data=(hist_valid, hand_valid),
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
    with open(
        os.path.join(PATHE_SAVE_MODEL, "model_hands_not_hands.yaml"), "w"
    ) as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    model.save_weights(os.path.join(PATHE_SAVE_MODEL, "model_hands_not_hands.h5"))
    print("OK")

    # evaluate the network
    Console.info("Evaluating network...")
    score = model.evaluate([hist_test], [hand_test], batch_size=BATCH_SIZE, verbose=1)

    Console.log("Test loss:", score[0])
    Console.log("Test Acc:", score[1])

    # list all data in history
    Console.info("Save model history graphics...")
    print(history.history.keys())


# Como vamos a usar multi procesos uno por core.
# Los procesos hijos cargan el mismo código.
# Este if permite que solo se ejecute lo que sigue si es llamado
# como proceso raíz.
if __name__ == "__main__":
    if args["predict"] == None or args["predict"] == "True":
        (X_train, y_train) = openDataSet()
        Console.log("Dataset file count", len(y_train))

    # Create model
    model = makerModel()

    if args["train"] == "True":
        trainModel(model, X_train, y_train)

    if args["evaluate"] == "True":
        Console.info("Evaluating model...")
        score = model.evaluate([X_train], [y_train], batch_size=BATCH_SIZE, verbose=1)
        Console.log("Test loss:", score[0])
        Console.log("Test Acc:", score[1])

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

            error_count = 0
            for i in range(len(ynew)):
                predict = 1 if ynew[i][0] > 0.5 else 0
                if y_train[i] != predict:
                    error_count = error_count + 1
                    Console.error(
                        "ID:", i,
                        "Original", y_train[i],
                        "Predict", predict,
                        "-", str(math.trunc(ynew[i][0]*100)) + "%"
                    )
            Console.wran("Img with error", error_count)
