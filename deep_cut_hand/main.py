#!/usr/bin/python3

# Cargar los dos modelos y sus pesos
# Elegir 1000 imagenes ala azar del dataset real
# Aplicar el modelo 1 para obtenes los valores del histograma
# corregir el histograma y evaluarlo con la otra red
# indicar que histograma es imagen y cuales no,
# para creae las iagenes en capetas separadas
# Depues evaluar los resultados
from keras.models import model_from_yaml
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, SGD
from multiprocessing import Process
from utilities import Console, updateProgress, getHistogram, histogramsLevelFix
import cv2
import fnmatch
import multiprocessing
import numpy as np
import os
import platform

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# https://keras.io/optimizers
# OPT = Adam(lr=0.001)
# OPT = RMSprop()
OPT = SGD(lr=0.01, clipvalue=0.5)
# OPT = Adadelta(lr=0.01, rho=0.95, epsilon=None, decay=0.0)
# OPT = Adagrad(lr=0.05)


# Use N images of dataset, If it is -1 using all dataset
CUT_DATASET = 1000

# For this problem the validation and test data provided by the concerned authority did not have labels,
# so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
# Directory of dataset to use
TRAIN_DIR = "boneage-training-dataset"
train_dir = os.path.join(__location__, "..", TRAIN_DIR)


def progressImg(files=[]):
    global img_file
    total_file = len(files)
    for i in range(total_file):
        (img_file, lower, upper) = files[i]
        # Update the progress bar
        progress = float(i / total_file), (i + 1)
        updateProgress(progress[0], progress[1], total_file, img_file)

        # Get image's path
        img_path = os.path.join(train_dir, img_file)
        # Read a image
        img = cv2.imread(img_path, 0)
    return histogramsLevelFix(img, lower, upper)


def getFiles():
    Console.info("Get imges form", TRAIN_DIR)
    # file names on train_dir
    files = os.listdir(train_dir)
    # filter image files
    files = [f for f in files if fnmatch.fnmatch(f, "*.png")]
    # Sort randomly
    np.random.shuffle(files)
    return files[:CUT_DATASET]


def loadModel(dir_model_backup):
    Console.info("Get model and weights for", dir_model_backup)
    # load YAML and create model
    yaml_file = open(os.path.join(__location__, "model", dir_model_backup + ".yaml"), "r")
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    model.load_weights(os.path.join(__location__, "model", dir_model_backup + ".h5"))
    Console.log("Loaded model from disk")
    return model


def getHistogramFormFiles(files=[]):
    total_file = len(files)
    Console.log("Process", total_file, "images")

    num_processes = multiprocessing.cpu_count()
    if platform.system() == "Linux" and num_processes > 1:
        processes = []

        lot_size = int(total_file / num_processes)

        for x in range(1, num_processes + 1):
            if x < num_processes:
                lot_img = files[(x - 1) * lot_size: ((x - 1) * lot_size) + lot_size]
            else:
                lot_img = files[(x - 1) * lot_size:]
            processes.append(Process(target=mpStart, args=(lot_img, output)))

        if len(processes) > 0:
            Console.info("Get histogram of the images...")
            for p in processes:
                p.start()

            result = []
            for x in range(num_processes):
                result.append(output.get(True))

            for p in processes:
                p.join()

            X_values = []
            for x in result:
                X_values = X_values + x
            updateProgress(1, total_file, total_file, "")
            Console.info("Image processed:", len(X_values))

    else:
        Console.info("No podemos dividir la cargan en distintos procesadores")
        X_values = progressImg(files)

    return X_values


# Usado en caso de usar multiples core
output = multiprocessing.Queue()


def mpStart(files, output):
    global img_file
    total_file = len(files)
    rta = []
    for i in range(total_file):
        img_file = files[i]
        # Update the progress bar
        progress = float(i / total_file), (i + 1)
        updateProgress(progress[0], progress[1], total_file, img_file)

        # Get image's path
        img_path = os.path.join(train_dir, img_file)
        # Read a image
        img = cv2.imread(img_path, 0)
        # Get histogram of hands
        rta.append((img_file, getHistogram(img)))

    output.put(rta)


if __name__ == "__main__":
    model_get_hand = loadModel("model_histogram")
    model_valid_hand = loadModel("model_hands_not_hands")

    # Read img files
    files = getFiles()
    # Get hist for hand
    X_hist_hands = getHistogramFormFiles(files)
    print(len(X_hist_hands))
