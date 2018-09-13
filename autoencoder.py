from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def readFile(gender, dataset, X_img=None, x_gender=None, y_age=None):
    print("Reading", gender, dataset, "data...")
    file_name = gender + "-" + dataset + "-" + ".hdf5"
    with h5py.File(
        os.path.join(__location__, "packaging-dataset", "for_autoencoder", file_name),
        "r+",
    ) as f:
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


######################################################################
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = np.reshape(
    x_train, (len(x_train), 28, 28, 1)
)  # adapt this if using `channels_first` image data format
x_test = np.reshape(
    x_test, (len(x_test), 28, 28, 1)
)  # adapt this if using `channels_first` image data format
######################################################################33
genderType = "male"
genderType = "female"
# x_test, _, _ = readFile(genderType, "testing")
# x_train, _, _ = readFile(genderType, "validation")
# input_img = Input(shape=x_train.shape[1:])

input_img = Input(shape=(28, 28, 1))
########################### Auto encoder ############################
x = Conv2D(16, (3, 3), activation="relu", padding="same", name="encoded_0")(input_img)
x = MaxPooling2D((2, 2), padding="same", name="mp2D_0")(x)
x = Conv2D(8, (3, 3), activation="relu", padding="same", name="encoded_1")(x)
x = MaxPooling2D((2, 2), padding="same", name="mp2D_1")(x)
x = Conv2D(8, (3, 3), activation="relu", padding="same", name="encoded_2")(x)
encoded = MaxPooling2D((2, 2), padding="same", name="mp2D_2")(x)

x = Conv2D(8, (3, 3), activation="relu", padding="same", name="decoded_3")(encoded)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional
x = UpSampling2D((2, 2), name="usl2D_0")(x)
x = Conv2D(8, (3, 3), activation="relu", name="decoded_2")(x)
x = UpSampling2D((2, 2), name="usl2D_1")(x)
x = Conv2D(16, (3, 3), activation="relu", name="decoded_1")(x)
x = UpSampling2D((2, 2), name="usl2D_2")(x)
decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="decoded_0")(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")
print(autoencoder.summary())

autoencoder.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test),
)

########################### Auto encoder ############################

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
