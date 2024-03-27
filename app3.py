# -*- coding: utf-8 -*-            
# @Author : zhangyong
# @Time : 2024/3/27 下午9:09
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QComboBox, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np


class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.filePath = ''  # 用于保存当前选中的文件路径
        self.output = None

    def initUI(self):
        self.setWindowTitle('刀具面积检测')
        self.setGeometry(100, 100, 800, 600)
        # self.setFixedSize(800, 600)
        mainLayout = QVBoxLayout()
        self.setLayout(mainLayout)
        # 图像展示区域
        self.imageLabel = QLabel()
        self.imageLabel.setAlignment(Qt.AlignCenter)  # 设置图片居中显示
        self.imageLabel.setFixedSize(580, 580)  # 根据需要调整大小
        mainLayout.addWidget(self.imageLabel, alignment=Qt.AlignCenter)
        # 操作和面积信息区域
        infoLayout = QHBoxLayout()
        self.label = QLabel('选择操作：')
        infoLayout.addWidget(self.label)
        self.comboBox = QComboBox()
        self.comboBox.addItems(['0', '500', '1000', '1500', '2000'])
        infoLayout.addWidget(self.comboBox)

        self.areaLabel = QLabel('面积：未知')
        infoLayout.addStretch()  # 添加弹性空间使面积标签向右对齐
        infoLayout.addWidget(self.areaLabel)

        mainLayout.addLayout(infoLayout)

        # 按钮区域
        buttonLayout = QHBoxLayout()
        self.selectButton = QPushButton('选择图片')
        self.selectButton.clicked.connect(self.selectImage)
        buttonLayout.addWidget(self.selectButton)

        self.processButton = QPushButton('处理图片')
        self.processButton.clicked.connect(self.processAndDisplayImage)
        buttonLayout.addWidget(self.processButton)

        self.saveButton = QPushButton('保存图片')
        self.saveButton.clicked.connect(self.saveImage)
        buttonLayout.addWidget(self.saveButton)

        mainLayout.addLayout(buttonLayout)

        # 调整布局和外边距
        mainLayout.setSpacing(10)
        infoLayout.setSpacing(10)
        buttonLayout.setSpacing(10)
        mainLayout.setContentsMargins(10, 10, 10, 10)
        infoLayout.setContentsMargins(200, 0, 200, 10)  # 调整信息区域的左右留白
        buttonLayout.setContentsMargins(10, 0, 10, 10)  # 调整按钮区域的左右留白

    def selectImage(self):
        options = QFileDialog.Options()
        self.filePath, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "All Files (*);;PNG Files (*.png);;JPG Files (*.jpg)", options=options)
        if self.filePath:
            pixmap = QPixmap(self.filePath)
            self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), aspectRatioMode=Qt.KeepAspectRatio))
            self.areaLabel.setText('面积：未知')

    def processAndDisplayImage(self):
        if self.filePath:
            img, total_area = self.processImage(self.filePath, self.comboBox.currentText())
            self.output = img
            self.displayImage(img)
            self.areaLabel.setText(f'总面积：{total_area}')

    def saveImage(self):
        if self.filePath:
            options = QFileDialog.Options()
            savePath, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "PNG Files (*.png);;JPG Files (*.jpg)", options=options)
            if savePath:
                # 假设`output`是处理后的图片，你应当在`processImage`方法中定义并更新这个变量
                cv2.imwrite(savePath, self.output)

    def processImage(self, filePath, option):
        # 这里根据option选择执行你的OpenCV代码，返回处理后的图像和面积
        # 示例代码，需要替换为你的实际处理代码
        img = cv2.imread(filePath)
        total_area = 0
        if option == '0':
            # 将图像数据转换为二维数组形式
            values = img.reshape((-1, 3))
            values = np.float32(values)
            # K-Means聚类
            K = 4
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # 创建新图像并根据聚类标签对像素点着色
            segmented_img = np.zeros_like(values)
            segmented_img[np.where(label == 0)[0], :] = [255, 0, 0]  # 给第0类像素点赋值蓝色
            segmented_img[np.where(label == 1)[0], :] = [0, 255, 0]  # 给第1类像素点赋值绿色
            segmented_img[np.where(label == 2)[0], :] = [0, 0, 255]  # 给第2类像素点赋值红色
            segmented_img[np.where(label == 3)[0], :] = [0, 255, 0]  # 给第3类像素点赋值黑色
            # segmented_img[np.where(label == 4)[0], :] = [255, 255, 255] # 给第3类像素点赋值白色
            # 将分割后图像重新转化成与原图像相同的维度
            segmented_img = segmented_img.reshape(img.shape)
            # 创建掩膜图像
            mask = np.zeros(segmented_img.shape[:2], dtype=np.uint8)
            mask[np.where(np.all(segmented_img == [0, 255, 0], axis=-1))] = 255
            # # 进行形态学开运算，去掉周围的绿色点
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # # 进行形态学闭运算，填充区域内部空隙
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            # 找到轮廓并获取最大轮廓及其面积
            contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # max_contour = max(contours, key=cv2.contourArea)
            for i, contour in enumerate(contours):
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                total_area += area
            # 绘制最大轮廓并显示在原图上
            output = img.copy()
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
            return output, total_area
        elif option == '500':
            # 将图像数据转换为二维数组形式
            values = img.reshape((-1, 3))
            values = np.float32(values)
            # K-Means聚类
            K = 7
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # 创建新图像并根据聚类标签对像素点着色
            segmented_img = np.zeros_like(values)
            segmented_img[np.where(label == 4)[0], :] = [0, 255, 0]
            segmented_img = segmented_img.reshape(img.shape)
            # 创建掩膜图像
            mask = np.zeros(segmented_img.shape[:2], dtype=np.uint8)
            mask[np.where(np.all(segmented_img == [0, 255, 0], axis=-1))] = 255
            # # 进行形态学开运算，去掉周围的绿色点
            kernel = np.ones((4, 3), np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            kernel = np.ones((30, 30), np.uint8)
            # # 进行形态学闭运算，填充区域内部空隙
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            # 找到轮廓并获取最大轮廓及其面积
            contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # max_contour = max(contours, key=cv2.contourArea)
            total_area = 0
            for i, contour in enumerate(contours):
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                total_area += area
            # 绘制最大轮廓并显示在原图上
            output = img.copy()
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
            return output, total_area
        elif option == '1000':
            # 将图像数据转换为二维数组形式
            values = img.reshape((-1, 3))
            values = np.float32(values)
            # K-Means聚类
            K = 7
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # 创建新图像并根据聚类标签对像素点着色
            segmented_img = np.zeros_like(values)
            segmented_img[np.where(label == 4)[0], :] = [0, 255, 0]  # 给第3类像素点赋值白色
            # 将分割后图像重新转化成与原图像相同的维度
            segmented_img = segmented_img.reshape(img.shape)
            # 创建掩膜图像
            mask = np.zeros(segmented_img.shape[:2], dtype=np.uint8)
            mask[np.where(np.all(segmented_img == [0, 255, 0], axis=-1))] = 255
            # # 进行形态学开运算，去掉周围的绿色点
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            kernel = np.ones((20, 20), np.uint8)
            # # 进行形态学闭运算，填充区域内部空隙
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            # 找到轮廓并获取最大轮廓及其面积
            contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_area = 0
            for i, contour in enumerate(contours):
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                total_area += area
            # 绘制最大轮廓并显示在原图上
            output = img.copy()
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
            return output, total_area
        elif option == '1500':
            # 将图像数据转换为二维数组形式
            values = img.reshape((-1, 3))
            values = np.float32(values)
            # K-Means聚类
            K = 4
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # 创建新图像并根据聚类标签对像素点着色
            segmented_img = np.zeros_like(values)
            segmented_img[np.where(label == 3)[0], :] = [0, 255, 0]  # 给第3类像素点赋值黑色
            # 将分割后图像重新转化成与原图像相同的维度
            segmented_img = segmented_img.reshape(img.shape)
            # 创建掩膜图像
            mask = np.zeros(segmented_img.shape[:2], dtype=np.uint8)
            mask[np.where(np.all(segmented_img == [0, 255, 0], axis=-1))] = 255
            # # 进行形态学开运算，去掉周围的绿色点
            kernel = np.ones((5, 5), np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            # # 进行形态学闭运算，填充区域内部空隙
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            # 找到轮廓并获取最大轮廓及其面积
            contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_area = 0
            for i, contour in enumerate(contours):
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                total_area += area
            # 绘制最大轮廓并显示在原图上
            output = img.copy()
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
            return output, total_area
        elif option == '2000':
            # 将图像数据转换为二维数组形式
            values = img.reshape((-1, 3))
            values = np.float32(values)
            # K-Means聚类
            K = 5
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, label, center = cv2.kmeans(values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            # 创建新图像并根据聚类标签对像素点着色
            segmented_img = np.zeros_like(values)
            segmented_img[np.where(label == 2)[0], :] = [0, 255, 0]  # 给第2类像素点赋值红色
            # 将分割后图像重新转化成与原图像相同的维度
            segmented_img = segmented_img.reshape(img.shape)
            # 创建掩膜图像
            mask = np.zeros(segmented_img.shape[:2], dtype=np.uint8)
            mask[np.where(np.all(segmented_img == [0, 255, 0], axis=-1))] = 255
            # # 进行形态学开运算，去掉周围的绿色点
            kernel = np.ones((8, 7), np.uint8)
            opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            kernel = np.ones((1, 1), np.uint8)
            # # 进行形态学闭运算，填充区域内部空隙
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            # 找到轮廓并获取最大轮廓及其面积
            contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_area = 0
            for i, contour in enumerate(contours):
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                total_area += area
            # 绘制最大轮廓并显示在原图上
            output = img.copy()
            cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
            return output, total_area
    def displayImage(self, img):
        # 将OpenCV图像转换为QPixmap显示在Label上
        qFormat = QImage.Format_Indexed8
        if len(img.shape) == 3:  # channels==3
            if img.shape[2] == 4:
                qFormat = QImage.Format_RGBA8888
            else:
                qFormat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qFormat)
        outImage = outImage.rgbSwapped()
        self.imageLabel.setPixmap(QPixmap.fromImage(outImage))
        self.imageLabel.adjustSize()
        #调整此方法以适应固定大小的展示
        # 将OpenCV图像转换为QPixmap显示在Label上
        # height, width, channel = img.shape
        # bytesPerLine = 3 * width
        # qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        #
        # # 确保图片保持比例且居中显示在固定大小的QLabel中
        # pixmap = QPixmap.fromImage(qImg)
        # scaledPixmap = pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        # self.imageLabel.setPixmap(scaledPixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())
