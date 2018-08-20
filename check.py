#!/usr/bin/python3

# Check metrics using trained weight files
from keras.models import model_from_yaml
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad
import h5py
import numpy as np
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

BATCH_SIZE = 35
# https://keras.io/optimizers
OPTIMIZER = Adam(lr=0.001)
# OPTIMIZER = RMSprop()
# OPTIMIZER = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# OPTIMIZER = Adagrad(lr=0.05)

# Choose gender to use
GENDER_TYPE = "famale_and_male"
# GENDER_TYPE = "famale"
# GENDER_TYPE = "male"

# Path to save model
PATH_SAVE_MODEL = os.path.join(__location__, "model_backup", GENDER_TYPE)


def readFile(gender, dataset, X_img=None, x_gender=None, y_age=None):
    print("Reading", gender, dataset, "data...")
    file_name = gender + "-" + dataset + "-" + ".hdf5"
    with h5py.File(os.path.join(__location__, "packaging-dataset", file_name), "r+") as f:
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
if GENDER_TYPE == "famale_and_male" or GENDER_TYPE == "famale":
    genderType = "famale"
    img_train, gdr_train, age_train = readFile(genderType, "training")
    img_valid, gdr_valid, age_valid = readFile(genderType, "validation")
    img_test, gdr_test, age_test = readFile(genderType, "testing")

if GENDER_TYPE == "famale_and_male" or GENDER_TYPE == "male":
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
print("img_valid shape:", img_valid.shape)
print("img_test shape:", img_test.shape)

img_final = np.concatenate(img_train, img_valid, img_test)
gdr_final = np.concatenate(gdr_train, gdr_valid, gdr_test)
age_final = np.concatenate(age_train, age_valid, age_test)

# load YAML and create model
yaml_file = open(os.path.join(PATH_SAVE_MODEL, "model.yaml"), "r")
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights(os.path.join(PATH_SAVE_MODEL, "model.h5"))
print("Loaded model from disk")

# printing a model summary to check what we constructed
print(model.summary())

# I think there is another way to do this
input_values = img_final
try:
    if model.get_layer(name="gdr_input") is not None:
        print("Model with gender")
        input_values = [img_final, gdr_final]
except:
    pass

# evaluate loaded model on test data
model.compile(optimizer=OPTIMIZER, loss="mean_squared_error", metrics=["MAE"])
score = model.evaluate(input_values, age_final, batch_size=BATCH_SIZE, verbose=1)

print("\nTest loss:", score[0])
print("Test MAE:", score[1])
