from get_dataset import openDataSet
from utilities import Console
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad
from keras.models import Model
import h5py
import matplotlib.pyplot as plt
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

EPOCHS = 15
BATCH_SIZE = 12
# https://keras.io/optimizers
OPTIMIZER = Adam(lr=0.001, amsgrad=True)
# OPTIMIZER = RMSprop()
# OPTIMIZER = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# OPTIMIZER = Adagrad(lr=0.05)

########################### Auto encoder ############################
def autoencoder(input_img):
    # output: (224x224)/24.5 input: 224x224
    x = Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu")(input_img)
    x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)  # 112x112x2048
    # output: 2048x2 input: 56x56x4096
    x = Conv2D(516, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)  # 28x28x4096
    # output: 4096x2 input: 28x28x8192
    encoded = Conv2D(1024, kernel_size=(3, 3), activation="relu", padding="same")(x)

    # output: 4096x2 input: 28x28x8192
    x = Conv2D(1024, kernel_size=(3, 3), activation="relu", padding="same")(encoded)
    x = UpSampling2D(size=(2, 2))(x)  # 56x56x8192
    # output: 2048x2 input: 56x56x4096
    x = Conv2D(516, kernel_size=(3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D(size=(2, 2))(x)  # 112x112x4096
    # output: 224x224 input: 224x224x1
    decoded = Conv2D(3, kernel_size=(3, 3), padding="same", activation="sigmoid")(x)
    return decoded


####################################################################
X_train, y_train = openDataSet("testing")
X_valid, y_valid = openDataSet("validation")
X_test, y_test = openDataSet("training")

input_img = Input(shape=X_train.shape[1:])

autoencoder = Model(input_img, autoencoder(input_img))
print(autoencoder.summary())
autoencoder.compile(optimizer=OPTIMIZER, loss="binary_crossentropy")
# autoencoder.compile(optimizer=OPTIMIZER,loss="mean_squared_error")

autoencoder_train = autoencoder.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_valid, y_valid),
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
plt.showow(bbox_inches="tight")

decoded_imgs = autoencoder.predict(X_test[:12])

n = 10
plt.figure()
for i in range(1, n + 1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display real output image
    ax = plt.subplot(2, n, i + n)
    plt.imshow(y_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show(bbox_inches="tight")
