# import the necessary packages
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, startAt=0, metrics=["mean_absolute_error"]):
        # store the output path for the figure, the path to the JSON
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = os.path.sep.join([figPath, "metrics.json"])
        self.startAt = startAt
        self.metrics = metrics

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}
        # if the JSON history path exists, load the training history
        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())
                # check to see if a starting epoch was supplied
                if self.startAt > 0:
                    # loop over the entries in the history log and
                    # trim any entries that are past the starting
                    # epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][: self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
        # check to see if the training history should be serialized
        # to file
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()
        # ensure at least two epochs have passed before plotting
        # (epoch starts at zero)
        if len(self.H["loss"]) > 1:
            N = range(len(self.H["loss"]))
            # plot the training loss
            plt.style.use("ggplot")

            plt.figure(figsize=(10, 10))
            plt.plot(N, self.H["loss"], label="training")
            plt.plot(N, self.H["val_loss"], label="validatin")
            plt.title("Training Loss [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend(loc="upper right")
            # save the figure
            plt.savefig(
                os.path.sep.join([self.figPath, "history_loss.png"]),
                bbox_inches="tight",
            )
            plt.close()

            for metric in self.metrics:
                plt.figure(figsize=(10, 10))
                plt.plot(N, self.H[metric], label="training")
                plt.plot(N, self.H["val_" + metric], label="validation")
                plt.title(
                    "{} [Epoch {}]\n Train: {} - Val: {}".format(
                        metric,
                        len(self.H["loss"]),
                        np.min(self.H[metric]),
                        np.min(self.H["val_" + metric]),
                    )
                )
                plt.xlabel("Epoch #")
                plt.ylabel(metric)
                plt.legend(loc="upper right")
                # save the figure
                plt.savefig(
                    os.path.sep.join([self.figPath, metric + ".png"]),
                    bbox_inches="tight",
                )
                plt.close()
