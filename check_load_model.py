#!/usr/bin/python3

# Check metrics using trained weight files
from keras.models import model_from_yaml
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad
from six.moves import cPickle
import numpy as np
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

BATCH_SIZE = 35
VERBOSE = 1
# https://keras.io/optimizers
OPTIMIZER = Adam(lr=0.001)
# OPTIMIZER = RMSprop()
# OPTIMIZER = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# OPTIMIZER = Adagrad(lr=0.05)

# Path to save model
PATHE_SAVE_MODEL = os.path.join(__location__, "model-backup")

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


# load YAML and create model
yaml_file = open(os.path.join(PATHE_SAVE_MODEL, "model.yaml"), 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights(os.path.join(PATHE_SAVE_MODEL, "model.h5"))
print("Loaded model from disk")

# printing a model summary to check what we constructed
print(model.summary())

# evaluate loaded model on test data
model.compile(optimizer=OPTIMIZER, loss="mean_squared_error", metrics=["MAE", "accuracy"])
score = model.evaluate(
    [img_final, gdr_final], age_final, batch_size=BATCH_SIZE, verbose=VERBOSE
)

print("\nTest loss:", score[0])
print("Test MAE:", score[1])
print("Test accuracy:", score[2])
