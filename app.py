from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import sys

class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('K-Means Image Segmenter')
        layout = QVBoxLayout()

        # Image Label for Display
        self.imageLabel = QLabel()
        layout.addWidget(self.imageLabel)
        # 下拉列表用于选择代号
        self.comboCode = QComboBox()
        self.comboCode.addItems(['0', '500', '1000', '1500', '2000'])
        layout.addWidget(self.comboCode)

        # Select Image Button
        self.btnSelectImage = QPushButton('Select Image')
        self.btnSelectImage.clicked.connect(self.openFileDialog)
        layout.addWidget(self.btnSelectImage)

        # Process Image Button
        self.btnProcessImage = QPushButton('Process Image')
        self.btnProcessImage.clicked.connect(self.processImage)
        layout.addWidget(self.btnProcessImage)

        self.setLayout(layout)


    def openFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "All Files (*);;PNG Files (*.png);;JPG Files (*.jpg)", options=options)
        if fileName:
            self.imagePath = fileName
            self.displayImage(cv2.imread(fileName)) # 显示选择的图片

    def processImage(self):
        code = self.comboCode.currentText()
        # 根据代号设置参数并处理图像
        if code == '0':
            contourImage = self.convertImage(self.imagePath, 0, 4, [0, 1, 2, 3], (5, 5), (5, 5))
        elif code == '500':
            contourImage = self.convertImage(self.imagePath, 500, 7, [4], (4, 3), (30, 30))
        elif code == '1000':
            contourImage = self.convertImage(self.imagePath, 1000, 7, [4], (3, 3), (20, 20))
        elif code == '1500':
            contourImage = self.convertImage(self.imagePath, 1500, 4, [3], (5, 5), (5, 5))
        elif code == '2000':
            contourImage = self.convertImage(self.imagePath, 2000, 5, [2], (8, 7), (1, 1))
        self.displayImage(contourImage)

    def convertImage(self, imagePath, code, k, colorIndexes, openingKernelSize, closingKernelSize):
        img = cv2.imread(imagePath)
        values = img.reshape((-1, 3))
        values = np.float32(values)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        segmented_img = np.zeros_like(values)
        # for index in colorIndexes:
            # segmented_img[np.where(label == index)[0], :] = [0, 255, 0]
        if code == 0:
            segmented_img[np.where(label == 0)[0], :] = [255, 0, 0]  # 给第0类像素点赋值蓝色
            segmented_img[np.where(label == 1)[0], :] = [0, 255, 0]  # 给第1类像素点赋值绿色
            segmented_img[np.where(label == 2)[0], :] = [0, 0, 255]  # 给第2类像素点赋值红色
            segmented_img[np.where(label == 3)[0], :] = [0, 255, 0]  # 给第3类像素点赋值黑色
        else:
            segmented_img[np.where(label == colorIndexes)[0], :] = [0, 255, 0]
        segmented_img = segmented_img.reshape(img.shape)
        mask = np.zeros(segmented_img.shape[:2], dtype=np.uint8)
        mask[np.where(np.all(segmented_img == [0, 255, 0], axis=-1))] = 255

        opening_kernel = np.ones(openingKernelSize, np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)

        closing_kernel = np.ones(closingKernelSize, np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closing_kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = img.copy()
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

        return output

    def displayImage(self, image):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        self.imageLabel.setPixmap(QPixmap.fromImage(qImg))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())
