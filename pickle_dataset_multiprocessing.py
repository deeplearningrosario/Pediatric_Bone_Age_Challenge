#!/usr/bin/python3

from pickle_dataset import loadDataSet, saveDataSet, getFiles, checkPath
from multiprocessing import Process
import multiprocessing
import os
import platform

# For this problem the validation and test data provided by the concerned authority did not have labels,
# so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def mpStart(files, output):
    print("Processing female images...")
    (X_train, x_gender, y_age) = loadDataSet(female)
    saveDataSet("female", X_train, x_gender, y_age)


if __name__ == "__main__":
    # Create the directories to save the images
    checkPath()

    female, male = getFiles()

    num_processes = multiprocessing.cpu_count()
    if platform.system() == "Linux" and num_processes > 1:
        processes = []
        processes.append(Process(target=mpStart, args=(female)))
        processes.append(Process(target=mpStart, args=(male)))

        if len(processes) > 0:
            print("Loading data set...")
            for p in processes:
                p.start()
            for p in processes:
                p.join()
    else:
        print("No podemos dividir la cargan en distintos procesadores")
        exit(0)
