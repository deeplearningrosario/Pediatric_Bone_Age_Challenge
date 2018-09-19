from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad

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
