#!/usr/bin/python3

from pickle_dataset import loadDataSet, saveDataSet, getFiles, checkPath
from multiprocessing import Process
import multiprocessing
import os
import platform

# For this problem the validation and test data provided by the concerned authority did not have labels,
# so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Usado en caso de usar multiples core
output = multiprocessing.Queue()


def mpSaveFiles(gender, result):
    print("Join the results of the processes...")
    X_train = []
    x_gender = []
    y_age = []
    for X_train_mp, x_gender_mp, y_age_mp in result:
        X_train = X_train + X_train_mp
        x_gender = x_gender + x_gender_mp
        y_age = y_age + y_age_mp
    saveDataSet(gender, X_train, x_gender, y_age)


def mpStart(gender, arg, output):
    output.put((gender, loadDataSet(arg)))


if __name__ == "__main__":
    # Create the directories to save the images
    checkPath()

    print("Loading data set...")
    female, male = getFiles()

    num_processes = multiprocessing.cpu_count()
    if platform.system() == "Linux" and num_processes > 1:
        processes = []
        num_processes = int(num_processes / 2)

        lot_size = int(len(female) / num_processes)
        for x in range(1, num_processes + 1):
            if x < num_processes:
                lot_img = female[(x - 1) * lot_size : ((x - 1) * lot_size) + lot_size]
            else:
                lot_img = female[(x - 1) * lot_size :]
            processes.append(Process(target=mpStart, args=("female", lot_img, output)))

        lot_size = int(len(male) / num_processes)
        for x in range(1, num_processes + 1):
            if x < num_processes:
                lot_img = male[(x - 1) * lot_size : ((x - 1) * lot_size) + lot_size]
            else:
                lot_img = male[(x - 1) * lot_size :]
            processes.append(Process(target=mpStart, args=("male", lot_img, output)))

        if len(processes) > 0:
            print("Processing images...")
            for p in processes:
                p.start()

            result_female = []
            result_male = []
            print("Joining result...")
            for x in range(num_processes * 2):
                gender, data = output.get(True)
                if gender == "male":
                    result_male.append(data)
                if gender == "female":
                    result_female.append(data)

            for p in processes:
                p.join()

            print("Start saving process...")
            processes = []
            processes.append(
                Process(target=mpSaveFiles, args=("female", result_female))
            )
            processes.append(Process(target=mpSaveFiles, args=("male", result_male)))
            for p in processes:
                p.start()
            for p in processes:
                p.join()

    else:
        print("No podemos dividir la cargan en distintos procesadores")
        exit(0)
