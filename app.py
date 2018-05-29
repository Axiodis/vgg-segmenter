import sys
import cv2

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi
import qdarkstyle
from predictor import Predictor
from main_window import Ui_MainWindow
from multiprocessing import Pool, Process


class App(QMainWindow, Ui_MainWindow):
    SEGMENT_CLASSES = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cow',
                       'Dining Table', 'Dog', 'Horse', 'Motorbike', 'Person', 'Potted Plant', 'Sheep', 'Sofa', 'Train',
                       'TV']

    def __init__(self):

        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.comboBoxSegmentClass.addItems(self.SEGMENT_CLASSES)
        self.comboBoxSegmentClass.setCurrentIndex(14)

        self.class_to_segment = 15

        self.image = None
        self.predictor = Predictor()

        self._connect_signals()

        self.setWindowTitle('VggFcn')
        self.show()

    def _connect_signals(self):
        self.startButton.clicked.connect(self.start_webcam)
        self.stopButton.clicked.connect(self.stop_webcam)

        self.comboBoxSegmentClass.currentIndexChanged[str].connect(self.change_class_to_segment)

    def start_webcam(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def process_frame(self):
        self.image = cv2.flip(self.image, 1)
        predict_image = self.predictor.predict_image(self.image)

        for i in range(predict_image.shape[0]):
            for j in range(predict_image.shape[1]):
                if predict_image[i, j] != self.class_to_segment:
                    self.image[i, j] = [0, 0, 0]

        # self.image = self.image[...,::-1]

    def update_frame(self):
        ret, self.image = self.capture.read()

        self.process_frame()

        # p = Process(target=self.process_frame)
        # p.start()
        # p.join()

        self.display_image(self.image)

    def display_image(self, image):

        # height, width, channels = image.shape
        # bytesPerLine = channels * width
        # out_image = QImage(image, width, height, bytesPerLine, QImage.Format_RGB888)

        out_image = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        out_image = out_image.rgbSwapped()

        self.label.setPixmap(QPixmap.fromImage(out_image))
        self.label.setScaledContents(True)

    def stop_webcam(self):
        if self.timer:
            self.timer.stop()

        self.label.clear()

    def change_class_to_segment(self, segment_class_name):
        self.class_to_segment = self.comboBoxSegmentClass.currentIndex() + 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = App()
    sys.exit(app.exec_())
