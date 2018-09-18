#!/usr/bin/python3
# ./main.py --load_weight ./weight_file.h5

from autoencoder import encodedModel, decodedModel
from trainingmonitor import TrainingMonitor
from keras.applications import InceptionV3, ResNet50, Xception
from keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    regularizers,
    UpSampling2D,
)
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad
from six.moves import cPickle
import argparse
import h5py
import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-lw", "--load_weights", help="Path to the file weights")
ap.add_argument(
    "-d", "--dataset", default="packaging-dataset", help="path to input dataset"
)
args = vars(ap.parse_args())

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# network and training
EPOCHS = 40
BATCH_SIZE = 12
VERBOSE = 1
# https://keras.io/optimizers
OPTIMIZER = Adam(lr=0.001, amsgrad=True)
# OPTIMIZER = RMSprop()
# OPTIMIZER = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# OPTIMIZER = Adagrad(lr=0.05)

# Usar Xception after deconcoder output or only Encoder output
USING_OUTPUT_DENCODER = True

# Path to save model
PATH_SAVE_MODEL = os.path.join(__location__, "model_backup", "autoencoder_regression")

# Save weights after every epoch
if not os.path.exists(PATH_SAVE_MODEL):
    os.makedirs(PATH_SAVE_MODEL)


def readFile(gender, dataset, X_img=None, x_gender=None, y_age=None):
    print("Reading", gender, dataset, "data...")
    file_name = gender + "-" + dataset + ".hdf5"
    with h5py.File(os.path.join(__location__, args["dataset"], file_name), "r+") as f:
        f_img = f["img"][()]
        f_gender = f["gender"][()]
        f_age = f["age"][()]
        f.close()
    if X_img is None:
        X_img = f_img
    else:
        X_img = np.concatenate((X_img, f_img), axis=0)

    if x_gender is None:
        x_gender = f_gender
    else:
        x_gender = np.concatenate((x_gender, f_gender), axis=0)

    if y_age is None:
        y_age = f_age
    else:
        y_age = np.concatenate((y_age, f_age), axis=0)

    return X_img, x_gender, y_age


# Load data
print("...loading training data")
genderType = "female"
img_train, gdr_train, age_train = readFile(genderType, "training")
img_valid, gdr_valid, age_valid = readFile(genderType, "validation")
img_test, gdr_test, age_test = readFile(genderType, "testing")

genderType = "male"
img_train, gdr_train, age_train = readFile(
    genderType, "training", img_train, gdr_train, age_train
)
img_valid, gdr_valid, age_valid = readFile(
    genderType, "validation", img_valid, gdr_valid, age_valid
)
img_test, gdr_test, age_test = readFile(
    genderType, "testing", img_test, gdr_test, age_test
)

print("img_train shape:", img_train.shape)
print("gdr_train shape:", gdr_train.shape)
print("age_train shape:", age_train.shape)
print("img_valid shape:", img_valid.shape)
print("gdr_valid shape:", gdr_valid.shape)
print("age_valid shape:", age_valid.shape)
print("img_test shape:", img_test.shape)
print("gdr_test shape:", gdr_test.shape)
print("age_test shape:", age_test.shape)


# Create regression model
def regressionModel(inputs):
    # We stack dense layers and dropout layers to avoid overfitting after that
    x = Dense(1256, activation="relu")(inputs)

    x1 = Dropout(0.4)(x)
    x1 = Dense(1256, activation="relu")(x1)
    x2 = Dropout(0.4)(x)
    x2 = Dense(1256, activation="relu")(x2)
    x = keras.layers.concatenate([x1, x2])

    # kernel_regularizer=regularizers.l2(0.01),
    # activity_regularizer=regularizers.l1(0.01),
    x1 = Dropout(0.35)(x)
    x1 = Dense(240, activation="relu")(x1)
    x2 = Dropout(0.35)(x)
    x2 = Dense(240, activation="relu")(x2)
    x = keras.layers.concatenate([x1, x2])

    # and the final prediction layer as output (should be the main logistic regression layer)
    model = Dense(1, activation="relu", name="prediction")(x)
    return model


# Main model
#######################################

# input layer
image_input = Input(shape=img_train.shape[1:], name="image_input")

output_encoder = encodedModel(image_input)
output_decoder = decodedModel(output_encoder)
if not USING_OUTPUT_DENCODER:
    output_img = Xception(weights="imagenet")(output_decoder)
else:
    output_img = Flatten()(output_encoder)
    output_img = Dense(img_train.shape[1] * 2, activation="relu")(output_img)
    output_img = Dense(2048, activation="relu")(output_img)
    output_img = Dense(1024, activation="relu")(output_img)

# Gender input layer
gdr_input = Input(shape=(1,), name="gdr_input")
output_gdr = Dense(2, activation="relu")(gdr_input)

# Concatenating CNN output with sex_dense output after going through shared layer
x = keras.layers.concatenate([output_img, output_gdr])
predictions = regressionModel(x)

# Now that we have created a model structure we can define it
# this defines the model with two inputs and one output
model = Model(inputs=[image_input, gdr_input], outputs=[predictions, output_decoder])

# printing a model summary to check what we constructed
print(model.summary())
# Imagen summary model
plot_model(
    model, to_file=os.path.join(PATH_SAVE_MODEL, "summary_model.png"), show_shapes=True
)

# Load weight
if args["load_weights"] != None:
    print("Loading weights from", args["load_weights"])
    model.load_weights(args["load_weights"])

model.compile(
    optimizer=OPTIMIZER,
    loss="mean_squared_error",
    metrics=["MAE", "binary_crossentropy"],
)

################################### CallBacks ######################################
# Reduce learning rate
reduceLROnPlat = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.8, patience=3, verbose=1, min_lr=0.0001
)

csv_logger = keras.callbacks.CSVLogger(os.path.join(PATH_SAVE_MODEL, "training.csv"))

# Save model fit progress
PATH_TRAING_MONITOR = os.path.join(PATH_SAVE_MODEL, "training_monitor")
if not os.path.exists(PATH_TRAING_MONITOR):
    os.makedirs(PATH_TRAING_MONITOR)

callbacks = [
    TrainingMonitor(
        PATH_TRAING_MONITOR,
        metrics=[
            "prediction_mean_absolute_error",
            "decoder_output_binary_crossentropy",
        ],
    ),
    reduceLROnPlat,
    csv_logger,
]
####################################################################################

history = model.fit(
    [img_train, gdr_train],
    [age_train, img_train],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    shuffle=True,
    verbose=VERBOSE,
    validation_data=([img_valid, gdr_valid], [age_valid, img_valid]),
    callbacks=callbacks,
)

# serialize model to YAML
model_yaml = model.to_yaml()
with open(os.path.join(PATH_SAVE_MODEL, "model.yaml"), "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights(os.path.join(PATH_SAVE_MODEL, "model.h5"))
print("Saved model to disk")

score = model.evaluate(
    [img_test, gdr_test], age_test, batch_size=BATCH_SIZE, verbose=VERBOSE
)

print("\nTest loss:", score[0])
print("Test MAE:", score[1])

# Save all data in history
with open(os.path.join(PATH_SAVE_MODEL, "history.pkl"), "wb") as f:
    cPickle.dump(history.history, f)
f.close()

# list all data in history
print(history.history.keys())

# plot the training loss and accuracy
plt.style.use("ggplot")

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "test"], loc="upper right")
plt.savefig(os.path.join(PATH_SAVE_MODEL, "history_loss.png"), bbox_inches="tight")
plt.close()

plt.plot(history.history["mean_absolute_error"], label="mean")
plt.plot(history.history["val_mean_absolute_error"], label="val_mean")
plt.title("Training Absolute Error")
plt.xlabel("Epoch")
plt.ylabel("Absolute Error")
plt.legend(["train", "test"], loc="upper right")
plt.savefig(os.path.join(PATH_SAVE_MODEL, "history_mean.png"), bbox_inches="tight")
plt.close()
