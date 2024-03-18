import cv2
import numpy as np
import os
# Load the original image
image_path = 'imgs2/1-1500.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Validate if image is loaded
if image is None:
    raise ValueError("Image not loaded properly")

# Define the initial lower and upper gray value thresholds
lower_gray_threshold = 100  # Initial lower threshold
upper_gray_threshold = 200  # Initial upper threshold

# Create a function to process the image with adjustable gray thresholds
def process_image_with_thresholds(lower_threshold, upper_threshold):
    # Create a mask for the specified gray value range
    gray_mask = cv2.inRange(image, lower_threshold, upper_threshold)

    # Create a color image
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Set the color (green) where the mask is
    color_image[gray_mask != 0] = [0, 255, 0]

    return color_image

# Process the image with the initial thresholds
result_image = process_image_with_thresholds(lower_gray_threshold, upper_gray_threshold)

cv2.imshow('',result_image)
cv2.waitKey(0)
# Save the result image to file
# result_image_path = '/mnt/data/processed_image_with_thresholds.png'
# cv2.imwrite(result_image_path, result_image)
