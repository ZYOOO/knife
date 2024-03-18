import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_and_display_image(file_path):
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
    # Draw the contours
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    # Calculate the area of the contour and create a mask
    mask = np.zeros_like(gray)  # Create a mask of zeros the same size as the grayscale image
    area = 0
    for contour in contours:
        area += cv2.contourArea(contour)  # Calculate the contour area
        cv2.drawContours(mask, [contour], -1, 255, -1)  # Draw filled contour in mask

    # Display the images
    titles = ['Original Image', 'Binary Threshold', 'Mask', 'Contour']
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB), thresh, mask, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]

    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

    return area


# Run the function on an example file
example_file_path = './imgs/2.jpg'
area_example = process_and_display_image(example_file_path)
print(area_example)
