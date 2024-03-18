import cv2
import numpy as np


def process_and_display_image_cv(file_path):
    # Read the image
    image = cv2.imread(file_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold the image
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    contour_img = image.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

    # Calculate the area of the contour and create a mask
    mask = np.zeros_like(gray)  # Create a mask of zeros the same size as the grayscale image
    area = 0
    for contour in contours:
        area += cv2.contourArea(contour)  # Calculate the contour area
        cv2.drawContours(mask, [contour], -1, 255, -1)  # Draw filled contour in mask

    # Display each step in separate OpenCV windows
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)  # Wait for a key press to continue
    cv2.imshow('Gray Image', gray)
    cv2.waitKey(0)
    cv2.imshow('Gaussian Blur', blur)
    cv2.waitKey(0)
    cv2.imshow('Binary Threshold', thresh)
    cv2.waitKey(0)
    cv2.imshow('Contours', contour_img)
    cv2.waitKey(0)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)

    # Close all the windows
    cv2.destroyAllWindows()

    return area


# Example usage:
file_path = './imgs2/1-1500.png'
area = process_and_display_image_cv(file_path)
print(f'Area of the contour: {area}')
