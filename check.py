#!/usr/bin/python3

# Check metrics using trained weight files
from keras.applications import InceptionV3, ResNet50, Xception
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.optimizers import Adam, RMSprop
from six.moves import cPickle
import numpy as np
import keras
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# network and training
EPOCHS = 30
BATCH_SIZE = 35
VERBOSE = 1

# https://keras.io/optimizers
# OPTIMIZER = Adam()
OPTIMIZER = RMSprop()
# OPTIMIZER = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

# Image processing layer
# CNN = 'Xception'
CNN = 'IV3'
# CNN = 'RN50'

# Load data
print("...loading training data")
f = open((os.path.join(__location__, "data.pkl")), "rb")
img = cPickle.load(f)
f.close()

f = open((os.path.join(__location__, "data_age.pkl")), "rb")
age = cPickle.load(f)
f.close()

f = open((os.path.join(__location__, "data_gender.pkl")), "rb")
gender = cPickle.load(f)
f.close()

img = np.asarray(img, dtype=np.float32)
age = np.asarray(age)
gender = np.asarray(gender)

# this is to normalize x since RGB scale is [0,255]
img /= 255.

img_final = []
age_final = []
gdr_final = []

# Shuffle images and split into train, validation and test sets
random_no = np.random.choice(img.shape[0], size=img.shape[0], replace=False)
for i in random_no:
    img_final.append(img[i, :, :, :])
    age_final.append(age[i])
    gdr_final.append(gender[i])

img_final = np.asarray(img_final)
age_final = np.asarray(age_final)
gdr_final = np.asarray(gdr_final)

print("img_final shape:" + str(img_final.shape))
print("age_final shape:" + str(age_final.shape))
print("gdr_final shape:" + str(gdr_final.shape))

# First we need to create a model structure
# input layer
image_input = Input(shape=img_final.shape[1:], name="image_input")

if CNN == 'IV3':
    # Inception V3 layer with pre-trained weights from ImageNet
    # base_iv3_model = InceptionV3(include_top=False, weights="imagenet")
    base_iv3_model = InceptionV3(weights="imagenet")
    # Inception V3 output from input layer
    output_vgg16 = base_iv3_model(image_input)
    # flattening it #why?
    # flat_iv3 = Flatten()(output_vgg16)
elif CNN == 'RN50':
    # ResNet50 layer with pre-trained weights from ImageNet
    base_rn50_model = ResNet50(weights="imagenet")
    # ResNet50 output from input layer
    output_rn50 = base_rn50_model(image_input)
elif CNN == 'Xception':
    # Xception layer with pre-trained weights from ImageNet
    base_xp_model = Xception(weights="imagenet")
    # Xception output from input layer
    output_xp = base_xp_model(image_input)

# Gender input layer
gdr_input = Input(shape=(1,), name="gdr_input")
# Gender dense layer
gdr_dense = Dense(32, activation="relu")
# Gender dense output
output_gdr_dense = gdr_dense(gdr_input)

if CNN == 'IV3':
    # Concatenating iv3 output with sex_dense output after going through shared layer
    x = keras.layers.concatenate([output_vgg16, output_gdr_dense])
elif CNN == 'RN50':
    # Concatenating ResNet50 output with gender_dense output after going through shared layer
    x = keras.layers.concatenate([output_rn50, output_gdr_dense])
elif CNN == 'Xception':
    # Concatenating Xception output with gender_dense output after going through shared layer
    x = keras.layers.concatenate([output_xp, output_gdr_dense])

# We stack dense layers and dropout layers to avoid overfitting after that
x = Dense(1000, activation="relu")(x)
x = Dropout(0)(x)
x = Dense(1000, activation="relu")(x)
x = Dropout(0)(x)

# and the final prediction layer as output (should be the main logistic regression layer)
# predictions = Dense(1, activation='sigmoid', name='predictions')(x)
predictions = Dense(1)(x)

# Now that we have created a model structure we can define it
# this defines the model with two inputs and one output
model = Model(inputs=[image_input, gdr_input], outputs=predictions)

# printing a model summary to check what we constructed
print(model.summary())

model.compile(optimizer=OPTIMIZER, loss="mean_squared_error", metrics=["MAE", "accuracy"])
model.load_weights('model.h5')

score = model.evaluate(
    [img_final, gdr_final], age_final, batch_size=BATCH_SIZE, verbose=VERBOSE
)

print("\nTest loss:", score[0])
print("Test MAE:", score[1])
print("Test accuracy:", score[2])
