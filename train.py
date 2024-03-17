if True:
    from reset_random import reset_random

    reset_random()
import os
import shutil

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from model import build_ssae
from utils import CLASSES, plot, TrainingCallback


def get_data():
    print("[INFO] Loading Data")
    f_path = "Data/features/features.npy"
    l_path = "Data/features/labels.npy"
    return np.load(f_path), np.load(l_path)


def train():
    reset_random()
    x, y = get_data()

    print("[INFO] Splitting Train|Test Data")
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.3, shuffle=True, random_state=42
    )
    y_cat = to_categorical(y, len(CLASSES))
    print("[INFO] X Shape :: {0}".format(x.shape))
    print("[INFO] Train X Shape :: {0}".format(train_x.shape))
    print("[INFO] Test X Shape :: {0}".format(test_x.shape))

    model_dir = "model"
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    acc_loss_csv_path = os.path.join(model_dir, "acc_loss.csv")
    model_path = os.path.join(model_dir, "model.h5")
    training_cb = TrainingCallback(acc_loss_csv_path, model_dir)
    checkpoint = ModelCheckpoint(
        model_path,
        save_best_only=True,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        verbose=False,
    )

    model = build_ssae(x.shape[1])

    initial_epoch = 0
    if os.path.isfile(model_path) and os.path.isfile(acc_loss_csv_path):
        print("[INFO] Loading Pre-Trained Model :: {0}".format(model_path))
        model.load_weights(model_path)
        initial_epoch = len(pd.read_csv(acc_loss_csv_path))

    print("[INFO] Fitting Data")
    model.fit(
        x,
        y_cat,
        validation_data=(x, y_cat),
        batch_size=1024,
        epochs=50,
        verbose=0,
        initial_epoch=initial_epoch,
        callbacks=[training_cb, checkpoint],
    )

    train_prob = model.predict(train_x, verbose=False)
    train_pred = np.argmax(train_prob, axis=1).ravel()
    plot(train_y.astype(int), train_pred, train_prob, "results/Train")

    test_prob = model.predict(test_x, verbose=False)
    test_pred = np.argmax(test_prob, axis=1).ravel()
    plot(test_y.astype(int), test_pred, test_prob, "results/Test")


if __name__ == "__main__":
    train()
