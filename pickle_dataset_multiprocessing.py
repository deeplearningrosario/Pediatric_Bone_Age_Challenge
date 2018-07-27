#!/usr/bin/python3

from multiprocessing import Process
import fnmatch
import multiprocessing
import os
import pickle_dataset as pDataset
import platform

# For this problem the validation and test data provided by the concerned authority did not have labels,
# so the training data was split into train, test and validation sets
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Usado en caso de usar multiples core
output = multiprocessing.Queue()


def mpStart(files, output):
    output.put(pDataset.loadDataSet(files))


if __name__ == "__main__":
    # Create the directories to save the images
    if pDataset.SAVE_IMAGE_FOR_DEBUGGER:
        for folder in ["histograms_level_fix", "cut_hand", "render", "mask"]:
            if not os.path.exists(os.path.join(__location__, pDataset.TRAIN_DIR, folder)):
                os.makedirs(os.path.join(__location__, pDataset.TRAIN_DIR, folder))
    if pDataset.SAVE_RENDERS:
        if not os.path.exists(os.path.join(__location__, pDataset.TRAIN_DIR, "render")):
            os.makedirs(os.path.join(__location__, pDataset.TRAIN_DIR, "render"))

    # file names on train_dir
    files = os.listdir(pDataset.train_dir)
    # filter image files
    files = [f for f in files if fnmatch.fnmatch(f, "*.png")]
    total_file = len(files)
    print("Image total:", total_file)

    num_processes = multiprocessing.cpu_count()
    if platform.system() == "Linux" and num_processes > 1:
        processes = []

        lot_size = int(total_file / num_processes)

        for x in range(1, num_processes + 1):
            if x < num_processes:
                lot_img = files[(x - 1) * lot_size: ((x - 1) * lot_size) + lot_size]
            else:
                lot_img = files[x * lot_size:]
            print(x, len(lot_img))
            processes.append(Process(target=mpStart, args=(lot_img, output)))

        if len(processes) > 0:
            print("Loading data set...")
            for p in processes:
                p.start()

            result = []
            for x in range(num_processes):
                result.append(output.get(True))

            for p in processes:
                p.join()

            X_train = []
            y_age = []
            y_gender = []
            for mp_X_train, mp_y_age, mp_y_gender in result:
                X_train = X_train + mp_X_train
                y_age = y_age + mp_y_age
                y_gender = y_gender + mp_y_gender
            # TODO:FIXME creo que no procesa todas las imgenes
            print(len(X_train))
            saveData(X_train, y_age, y_gender)
    else:
        print("No podemos dividir la cargan en distintos procesadores")
        exit(0)
