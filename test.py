from model import build_ssae

if True:
    from reset_random import reset_random

    reset_random()
import os
import sys

import cmapy
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThreadPool
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QVBoxLayout,
    QGroupBox,
    QGridLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QLabel,
    QScrollArea,
    QDialog,
    QProgressBar,
)

from feature_extraction import (
    SHAPE,
    get_feature,
    get_feature_image,
    get_feature_map_model,
    get_image_to_predict, get_nasnet_large_model,
)
from preprocessing import preprocess
from utils import Worker, CLASSES


class MainGUI(QWidget):
    def __init__(self):
        super(MainGUI, self).__init__()

        self.setWindowTitle("Oral Cancer Detection")
        self.setWindowFlags(Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setWindowState(Qt.WindowMaximized)

        self.app_width = QApplication.desktop().availableGeometry().width()
        self.app_height = QApplication.desktop().availableGeometry().height()
        app.setFont(QFont("JetBrains Mono"))

        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignCenter | Qt.AlignTop)

        self.gb_1 = QGroupBox("Input Data")

        self.gb_1.setFixedWidth((self.app_width // 100) * 99)
        self.gb_1.setFixedHeight((self.app_height // 100) * 10)
        self.grid_1 = QGridLayout()
        self.grid_1.setSpacing(10)
        self.gb_1.setLayout(self.grid_1)

        self.ip_le = QLineEdit()
        self.ip_le.setFixedWidth((self.app_width // 100) * 30)
        self.ip_le.setFocusPolicy(Qt.NoFocus)
        self.grid_1.addWidget(self.ip_le, 0, 0)

        self.ci_pb = QPushButton("Choose Input Image")
        self.ci_pb.clicked.connect(self.choose_input)
        self.grid_1.addWidget(self.ci_pb, 0, 1)

        self.pp_btn = QPushButton("Preprocessing")
        self.pp_btn.clicked.connect(self.preprocess_thread)
        self.grid_1.addWidget(self.pp_btn, 0, 2)

        self.fe_btn = QPushButton("Feature Extraction NASNetLarge")
        self.fe_btn.clicked.connect(self.fe_thread)
        self.grid_1.addWidget(self.fe_btn, 0, 3)

        self.cls_btn = QPushButton("Classify SSAE")
        self.cls_btn.clicked.connect(self.classify_thread)
        self.grid_1.addWidget(self.cls_btn, 0, 4)

        self.gb_2 = QGroupBox("Results")
        self.gb_2.setFixedWidth((self.app_width // 100) * 99)
        self.gb_2.setFixedHeight((self.app_height // 100) * 85)
        self.grid_2_scroll = QScrollArea()
        self.gb_2_v_box = QVBoxLayout()
        self.grid_2_widget = QWidget()
        self.grid_2 = QGridLayout(self.grid_2_widget)
        self.grid_2_scroll.setWidgetResizable(True)
        self.grid_2_scroll.setWidget(self.grid_2_widget)
        self.gb_2_v_box.addWidget(self.grid_2_scroll)
        self.gb_2_v_box.setContentsMargins(0, 0, 0, 0)
        self.gb_2.setLayout(self.gb_2_v_box)

        self.main_layout.addWidget(self.gb_1)
        self.main_layout.addWidget(self.gb_2)
        self.setLayout(self.main_layout)
        self._input_image_path = ""
        self._image_size = (
            (self.gb_2.height() // 100) * 90,
            (self.app_width // 100) * 45,
        )
        self.index = 0
        self.pp_data = {}
        self.load_screen = Loading()
        self.thread_pool = QThreadPool()
        self.feature = None
        self.class_ = None
        self.cls = None
        self.disable()
        self.show()

    def choose_input(self):
        self.reset()
        filter_ = "JPG Files (*.jpg)"
        self._input_image_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Choose Input Image",
            directory="Data/source",
            options=QFileDialog.DontUseNativeDialog,
            filter=filter_,
        )
        if os.path.isfile(self._input_image_path):
            self.ip_le.setText(self._input_image_path)
            self.add_image(self._input_image_path, "Input Image")
            self.ci_pb.setEnabled(False)
            self.pp_btn.setEnabled(True)
        else:
            self.show_message_box(
                "InputImageError", QMessageBox.Critical, "Choose valid image?"
            )

    def preprocess_thread(self):
        worker = Worker(self.preprocess_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.preprocess_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()
        self.pp_btn.setEnabled(False)

    def preprocess_runner(self):
        self.pp_data = preprocess(self._input_image_path)

    def preprocess_finisher(self):
        for k in self.pp_data:
            cv2.imwrite("tmp.jpg", self.pp_data[k])
            self.add_image("tmp.jpg", k)
        self.load_screen.close()
        self.fe_btn.setEnabled(True)

    def fe_thread(self):
        worker = Worker(self.fe_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.fe_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()
        self.fe_btn.setEnabled(False)

    def fe_runner(self):
        nan = get_nasnet_large_model()
        nan_fmm = get_feature_map_model(nan)
        nan_im = get_image_to_predict("tmp.jpg")
        self.feature = get_feature(nan_im, nan)
        nan_fm = get_feature_image(nan_im, nan_fmm)
        nan_fm = cv2.resize(nan_fm, SHAPE[:-1])
        nan_fm = cv2.applyColorMap(nan_fm, cmapy.cmap("viridis_r"))
        cv2.imwrite("tmp.jpg", nan_fm)

    def fe_finisher(self):
        self.add_image("tmp.jpg", "Feature Map")
        self.load_screen.close()
        self.cls_btn.setEnabled(True)

    def classify_thread(self):
        worker = Worker(self.classify_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.classify_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()
        self.cls_btn.setEnabled(False)

    def classify_runner(self):
        reset_random()
        x = np.array([self.feature])
        model = build_ssae(x.shape[1])
        model.built = True
        model.load_weights("model/model.h5")
        prob = model.predict(x)
        pred = np.argmax(prob, axis=1)[0]
        self.class_ = "Classified As :: {0} ({1:.2f}%)".format(
            CLASSES[pred], prob[0][pred] * 100
        )

    def classify_finisher(self):
        self.add_image(self._input_image_path, self.class_)
        self.load_screen.close()
        self.ci_pb.setEnabled(True)
        os.remove("tmp.jpg")

    def segment_thread(self):
        worker = Worker(self.segment_runner)
        self.thread_pool.start(worker)
        worker.signals.finished.connect(self.segment_finisher)
        self.load_screen.setWindowModality(Qt.ApplicationModal)
        self.load_screen.show()
        self.seg_btn.setEnabled(False)

    @staticmethod
    def clear_layout(layout):
        while layout.count() > 0:
            item = layout.takeAt(0)
            if not item:
                continue
            w = item.widget()
            if w:
                w.deleteLater()

    @staticmethod
    def show_message_box(title, icon, msg):
        msg_box = QMessageBox()
        msg_box.setFont(QFont("JetBrains Mono", 10, 1))
        msg_box.setWindowTitle(title)
        msg_box.setText(msg)
        msg_box.setIcon(icon)
        msg_box.setDefaultButton(QMessageBox.Ok)
        msg_box.setWindowModality(Qt.ApplicationModal)
        msg_box.exec_()

    def add_image(self, im_path, title):
        image_lb = QLabel()
        image_lb.setFixedHeight(self._image_size[0])
        image_lb.setFixedWidth(self._image_size[1])
        image_lb.setScaledContents(True)
        image_lb.setStyleSheet("padding-top: 30px;")
        qimg = QImage(im_path)
        pixmap = QPixmap.fromImage(qimg)
        image_lb.setPixmap(pixmap)
        self.grid_2.addWidget(image_lb, 0, self.index, Qt.AlignCenter)
        txt_lb = QLabel(title)
        self.grid_2.addWidget(txt_lb, 1, self.index, Qt.AlignCenter)
        self.index += 1

    def disable(self):
        self.ip_le.clear()
        self._input_image_path = ""
        self.pp_btn.setEnabled(False)
        self.fe_btn.setEnabled(False)
        self.cls_btn.setEnabled(False)
        self.feature = None
        self.class_ = None
        self.cls = None
        self.pp_data = {}

    def reset(self):
        self.disable()
        self.clear_layout(self.grid_2)


class Loading(QDialog):
    def __init__(self, parent=None):
        super(Loading, self).__init__(parent)
        self.screen_size = app.primaryScreen().size()
        self._width = int(self.screen_size.width() / 100) * 40
        self._height = int(self.screen_size.height() / 100) * 5
        self.setGeometry(0, 0, self._width, self._height)
        x = (self.screen_size.width() - self.width()) // 2
        y = (self.screen_size.height() - self.height()) // 2
        self.move(x, y)
        self.setWindowFlags(Qt.CustomizeWindowHint)
        self.pb = QProgressBar(self)
        self.pb.setFixedWidth(self.width())
        self.pb.setFixedHeight(self.height())
        self.pb.setRange(0, 0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    builder = MainGUI()
    sys.exit(app.exec_())
