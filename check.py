#!/usr/bin/python3

# Check metrics using trained weight files
from keras.models import model_from_yaml
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

BATCH_SIZE = 36
# https://keras.io/optimizers
OPTIMIZER = Adam(lr=0.001, amsgrad=True)
# OPTIMIZER = RMSprop()
# OPTIMIZER = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
# OPTIMIZER = Adagrad(lr=0.05)

# Choose gender to use
GENDER_TYPE = "female_and_male"
# GENDER_TYPE = "female"
# GENDER_TYPE = "male"

# Run evaluate model with all data set
EVALUATE_WITH_ALL_DATA_SET = False

# Path to save model
PATH_SAVE_MODEL = os.path.join(__location__, "model_backup", GENDER_TYPE)

# Show result on terminal
SHOW_PREDICT_DATA = False


def generate_graph(x_img, x_gdr, y_age, title):
    # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
    plt.style.use("ggplot")

    # I think there is another way to do this
    input_values = x_img
    try:
        if model.get_layer(name="gdr_input") is not None:
            input_values = [x_img, x_gdr]
    except:
        pass

    print("\nEvaluate model for", title.lower())
    score = model.evaluate(input_values, y_age, batch_size=BATCH_SIZE, verbose=1)
    print("loss:", score[0])
    print("MAE:", score[1])

    print("Generate graphics...")
    print("Predicted", title.lower())

    # make a prediction
    y_new = model.predict(input_values)

    if SHOW_PREDICT_DATA:
        print("Predict for test data")
        for i in range(len(y_new)):
            print("ID:", i, "Original:", y_age[i], "Predict:", y_new[i][0])

    # summarize history for mean
    _, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.plot(y_age, y_new, "r.", label="predictions")
    ax1.plot(y_age, y_age, "b.", label="actual")
    ax1.set_title(title)
    ax1.set_xlabel("Actual Age (Months)")
    ax1.set_ylabel("Predicted Age (Months)")
    ax1.legend(["Real", "Predict"], loc="upper left")
    plt.savefig(
        os.path.join(
            PATH_SAVE_MODEL,
            "predicted_" + title.lower().replace(" ", "_") + "_graph_1.png",
        ),
        bbox_inches="tight",
    )
    plt.show()

    _, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.plot(y_age, y_new, "r.", label="predictions")
    ax1.plot(y_age, y_age, "b-", label="actual")
    ax1.set_title(title)
    ax1.set_xlabel("Actual Age (Months)")
    ax1.set_ylabel("Predicted Age (Months)")
    ax1.legend(["Real", "Predict"], loc="upper left")
    plt.savefig(
        os.path.join(
            PATH_SAVE_MODEL,
            "predicted_" + title.lower().replace(" ", "_") + "_graph_1b.png",
        ),
        bbox_inches="tight",
    )
    plt.close()

    _, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.plot(y_new, "r.", label="predictions")
    ax1.plot(y_age, "b.", label="actual")
    ax1.set_title(title)
    ax1.set_xlabel("Actual Age (Months)")
    ax1.set_ylabel("Predicted Age (Months)")
    ax1.legend(["Real", "Predict"], loc="upper left")
    plt.savefig(
        os.path.join(
            PATH_SAVE_MODEL,
            "predicted_" + title.lower().replace(" ", "_") + "_graph_2.png",
        ),
        bbox_inches="tight",
    )
    plt.close()


def readFile(gender, dataset, X_img=None, x_gender=None, y_age=None):
    print("Reading", gender, dataset, "data...")
    file_name = gender + "-" + dataset + "-" + ".hdf5"
    with h5py.File(
        os.path.join(__location__, "packaging-dataset", file_name), "r+"
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


# Load data
print("...loading training data")
if GENDER_TYPE == "female_and_male" or GENDER_TYPE == "female":
    genderType = "female"
    img_train, gdr_train, age_train = readFile(genderType, "training")
    img_valid, gdr_valid, age_valid = readFile(genderType, "validation")
    img_test, gdr_test, age_test = readFile(genderType, "testing")

if GENDER_TYPE == "female_and_male" or GENDER_TYPE == "male":
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

print("Joining train, valid and test dataset.")
img_final = np.concatenate((img_train, img_valid, img_test), axis=0)
gdr_final = np.concatenate((gdr_train, gdr_valid, gdr_test), axis=0)
age_final = np.concatenate((age_train, age_valid, age_test), axis=0)

print("Loading model from disk")
# load YAML and create model
yaml_file = open(os.path.join(PATH_SAVE_MODEL, "model.yaml"), "r")
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)

# printing a model summary to check what we constructed
print(model.summary())
print("Model for", GENDER_TYPE)

# load weights into new model
model.load_weights(os.path.join(PATH_SAVE_MODEL, "model.h5"))

# evaluate loaded model on test data
model.compile(optimizer=OPTIMIZER, loss="mean_squared_error", metrics=["MAE"])

if EVALUATE_WITH_ALL_DATA_SET:
    generate_graph(img_final, gdr_final, age_final, "Boone age for all data set")

generate_graph(img_test, gdr_test, age_test, "Boone age for test data set")
generate_graph(img_train, gdr_train, age_train, "Boone age for train data set")
generate_graph(img_valid, gdr_valid, age_valid, "Boone age for valid data set")
