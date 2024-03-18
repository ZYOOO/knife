from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QLineEdit
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

        # K Value Input
        self.kInput = QLineEdit()
        self.kInput.setPlaceholderText('Enter K value for clustering')
        layout.addWidget(self.kInput)

        # Process Image Button for Clustering
        self.btnProcessClustering = QPushButton('Cluster Image')
        self.btnProcessClustering.clicked.connect(self.processClustering)
        layout.addWidget(self.btnProcessClustering)

        # Color Index Input
        self.colorIndexInput = QLineEdit()
        self.colorIndexInput.setPlaceholderText('Enter color index to highlight')
        layout.addWidget(self.colorIndexInput)

        # Highlight Cluster Button
        self.btnHighlightCluster = QPushButton('Highlight Cluster')
        self.btnHighlightCluster.clicked.connect(self.highlightCluster)
        layout.addWidget(self.btnHighlightCluster)

        # Select Image Button
        self.btnSelectImage = QPushButton('Select Image')
        self.btnSelectImage.clicked.connect(self.openFileDialog)
        layout.addWidget(self.btnSelectImage)

        self.setLayout(layout)

    def openFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "All Files (*);;PNG Files (*.png);;JPG Files (*.jpg)", options=options)
        if fileName:
            self.imagePath = fileName
            self.originalImage = cv2.imread(fileName)  # Store the original image for later use
            self.displayImage(self.originalImage)  # Display the selected image

    def processClustering(self):
        k = int(self.kInput.text())
        self.clusteredImage, self.labels = self.clusterImage(self.imagePath, k)
        self.displayImage(self.clusteredImage)

    def clusterImage(self, imagePath, k):
        img = cv2.imread(imagePath)
        self.originalImage = img.copy()  # Save the original image for later use
        values = img.reshape((-1, 3))
        values = np.float32(values)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Map each pixel to its corresponding cluster color
        centers = np.uint8(centers)
        clustered_img = centers[labels.flatten()]
        clustered_img = clustered_img.reshape((img.shape))

        return clustered_img, labels.reshape(img.shape[:2])

    def highlightCluster(self):
        if hasattr(self, 'labels'):  # Check if clustering was done
            colorIndex = int(self.colorIndexInput.text())
            highlightedImage, totalArea = self.markCluster(self.originalImage, self.labels, colorIndex)
            self.displayImage(highlightedImage)
            print(f"Total area of the cluster {colorIndex}: {totalArea} pixels")
        else:
            print("Please cluster the image before highlighting.")

    def markCluster(self, originalImage, labels, colorIndex):
        mask = np.zeros(labels.shape, dtype=np.uint8)
        mask[labels == colorIndex] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = originalImage.copy()
        cv2.drawContours(output, contours, -1, (0, 255, 0), 3)  # Draw the selected cluster with green contours

        totalArea = sum(cv2.contourArea(contour) for contour in contours)
        return output, totalArea

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
