from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad
from keras.models import Model
from trainingmonitor import TrainingMonitor
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

EPOCHS = 30
BATCH_SIZE = 10
# https://keras.io/optimizers
# OPTIMIZER = Adam(lr=0.001, amsgrad=True)
# OPTIMIZER = RMSprop()
OPTIMIZER = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# OPTIMIZER = Adagrad(lr=0.05)

ap = argparse.ArgumentParser()
ap.add_argument("-lw", "--load_weights", help="Path to the file weights")
ap.add_argument(
    "-d", "--dataset", default="packaging-dataset", help="path to input dataset"
)
args = vars(ap.parse_args())


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


########################### Auto encoder ############################
def encodedModel(inputs):
    x = Conv2D(
        1024, kernel_size=(3, 3), padding="same", activation="relu", name="encoder_1"
    )(inputs)
    x = MaxPooling2D(pool_size=(4, 4), padding="same", name="encoder_2")(x)
    x = Conv2D(
        256, kernel_size=(3, 3), activation="relu", padding="same", name="encoder_3"
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="encoder_4")(x)
    x = Conv2D(
        128, kernel_size=(3, 3), activation="relu", padding="same", name="encoder_5"
    )(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same", name="encoder_6")(x)
    encoded = Conv2D(
        64, kernel_size=(3, 3), activation="relu", padding="same", name="encoded_output"
    )(x)
    return encoded


def decodedModel(inputs):
    x = Conv2D(
        64, kernel_size=(3, 3), activation="relu", padding="same", name="decoder_1"
    )(inputs)
    x = UpSampling2D(size=(2, 2), name="decoder_2")(x)
    x = Conv2D(
        128, kernel_size=(3, 3), activation="relu", padding="same", name="decoder_3"
    )(x)
    x = UpSampling2D(size=(2, 2), name="decoder_4")(x)
    x = Conv2D(
        256, kernel_size=(3, 3), activation="relu", padding="same", name="decoder_5"
    )(x)
    x = UpSampling2D(size=(4, 4), name="decoder_6")(x)
    decoded = Conv2D(
        3,
        kernel_size=(3, 3),
        padding="same",
        activation="sigmoid",
        name="decoder_output",
    )(x)
    return decoded


####################################################################


# Run presses if this file is main
if __name__ == "__main__":
    # Path to save model
    PATH_SAVE_MODEL = os.path.join(__location__, "model_backup", "autoencoder")
    # Save model fit progress
    PATH_TRAING_MONITOR = os.path.join(PATH_SAVE_MODEL, "training_monitor")
    if not os.path.exists(PATH_TRAING_MONITOR):
        os.makedirs(PATH_TRAING_MONITOR)

    genderType = "female"
    x_train, _, _ = readFile(genderType, "training")
    x_valid, _, _ = readFile(genderType, "validation")
    x_test, _, _ = readFile(genderType, "testing")

    genderType = "male"
    x_train, _, _ = readFile(genderType, "training", x_train)
    x_valid, _, _, = readFile(genderType, "validation", x_valid)
    x_test, _, _ = readFile(genderType, "testing", x_test)

    input_img = Input(shape=x_train.shape[1:])

    output_encoder = encodedModel(input_img)
    output_decoder = decodedModel(output_encoder)

    autoencoder = Model(inputs=[input_img], outputs=[output_decoder])
    print(autoencoder.summary())

    autoencoder.compile(optimizer=OPTIMIZER, loss="binary_crossentropy")

    autoencoder_train = autoencoder.fit(
        x_train,
        x_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(x_valid, x_valid),
        callbacks=[TrainingMonitor(PATH_TRAING_MONITOR, metrics=[])],
    )

    loss = autoencoder_train.history["loss"]
    val_loss = autoencoder_train.history["val_loss"]
    epochs = range(EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()

    decoded_imgs = autoencoder.predict(x_test[:12])

    n = 10
    plt.figure()
    for i in range(1, n + 1):
        # display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show(bbox_inches="tight")
