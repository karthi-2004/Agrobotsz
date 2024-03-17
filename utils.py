import os
import sys
import traceback

import matplotlib
import pandas as pd
import prettytable
from PyQt5.QtCore import QRunnable, pyqtSignal, QObject, pyqtSlot
from keras.callbacks import Callback
from matplotlib import pyplot as plt
from performance_evaluator.metrics import evaluate
from performance_evaluator.plots import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

plt.set_cmap("Dark2")
plt.rcParams["font.family"] = "JetBrains Mono"
matplotlib.use("Qt5Agg")

CLASSES = ['Normal', 'OSCC']

GRAPHS = {
    "Train": {
        "CONF_MAT": plt.figure(),
        "PR_CURVE": plt.figure(),
        "ROC_CURVE": plt.figure(),
    },
    "Test": {
        "CONF_MAT": plt.figure(),
        "PR_CURVE": plt.figure(),
        "ROC_CURVE": plt.figure(),
    },
}


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            print(e)
            traceback.print_exc()
            exc_type, value = sys.exc_info()[:2]
            self.signals.error.emit((exc_type, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


def print_df_to_table(df, p=True):
    field_names = list(df.columns)
    p_table = prettytable.PrettyTable(field_names=field_names)
    p_table.add_rows(df.values.tolist())
    d = "\n".join(
        ["\t\t{0}".format(p_) for p_ in p_table.get_string().splitlines(keepends=False)]
    )
    if p:
        print(d)
    return d


class TrainingCallback(Callback):
    def __init__(self, acc_loss_path, name):
        self.acc_loss_path = acc_loss_path
        self.name = name
        if os.path.isfile(self.acc_loss_path):
            self.df = pd.read_csv(self.acc_loss_path)
        else:
            self.df = pd.DataFrame(
                [], columns=["epoch", "accuracy", "val_accuracy", "loss", "val_loss"]
            )
            self.df.to_csv(self.acc_loss_path, index=False)
        Callback.__init__(self)

    def on_epoch_end(self, epoch, logs=None):
        self.df.loc[len(self.df.index)] = [
            int(epoch + 1),
            round(logs["accuracy"], 4),
            round(logs["val_accuracy"], 4),
            round(logs["loss"], 4),
            round(logs["val_loss"], 4),
        ]
        self.df.to_csv(self.acc_loss_path, index=False)
        print(
            "[EPOCH :: {0}] -> Acc :: {1} | Val_Acc :: {2} | Loss :: {3} | Val_Loss :: {4}".format(
                epoch + 1, *[format(v, ".4f") for v in self.df.values[-1][1:]]
            )
        )
        plot_acc_loss(self.df, self.name)


def plot_line(y1, y2, epochs, for_, save_path):
    fig = plt.figure(num=1)
    plt.plot(range(epochs), y1, label="Training", color="dodgerblue")
    plt.plot(range(epochs), y2, label="Validation", color="orange")
    plt.title("Training and Validation {0}".format(for_))
    plt.xlabel("Epochs")
    plt.ylabel(for_)
    plt.xlim([0, epochs])
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()
    plt.close(fig)


def plot_acc_loss(df, name):
    epochs = len(df)
    acc = df["accuracy"].values
    val_acc = df["val_accuracy"].values
    loss = df["loss"].values
    val_loss = df["val_loss"].values
    plot_line(acc, val_acc, epochs, "Accuracy", "{0}/accuracy.png".format(name))
    plot_line(loss, val_loss, epochs, "Loss", "{0}/loss.png".format(name))


def plot(y, pred, prob, results_dir):
    for_ = os.path.basename(results_dir)
    print("[INFO] Evaluating {0} Data".format(for_))
    os.makedirs(results_dir, exist_ok=True)

    m = evaluate(y, pred, prob, CLASSES)
    df = m.overall_metrics
    df.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)
    print_df_to_table(df)

    fig = GRAPHS[for_]["CONF_MAT"]
    ax = fig.gca()
    ax.clear()
    confusion_matrix(
        y, pred, CLASSES, ax=ax, yticklabels_rotation=0
    )
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "conf_mat.png"))

    fig = GRAPHS[for_]["PR_CURVE"]
    ax = fig.gca()
    ax.clear()
    precision_recall_curve(y, prob, CLASSES, ax=ax, legend_ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "pr_curve.png"))

    fig = GRAPHS[for_]["ROC_CURVE"]
    ax = fig.gca()
    ax.clear()
    roc_curve(y, prob, CLASSES, ax=ax, legend_ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "roc_curve.png"))
