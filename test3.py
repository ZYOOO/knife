import cv2
import numpy as np

def process_image(file_path):
    # Read the image in grayscale
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Apply a Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use Otsu's thresholding method to binarize the processed image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find the contours in the image
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours based on a heuristic relative to the image size
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > image.size * 0.001]

    # Draw contours on the original image
    contour_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for display
    cv2.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 3)

    # Display each step
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)
    cv2.imshow('Blurred Image', blurred)
    cv2.waitKey(0)
    cv2.imshow('Binary Image', binary)
    cv2.waitKey(0)
    cv2.imshow('Opening', opening)
    cv2.waitKey(0)
    cv2.imshow('Contours', contour_img)
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()

    # Save and return the path of the contour image
    contour_img_path = file_path.replace('.png', '_contours.png')
    cv2.imwrite(contour_img_path, contour_img)

    return contour_img_path

# Example usage:
# Replace 'path_to_image.png' with your actual file path
contour_img_path = process_image('./imgs/2.jpg')
print(f"Contour image saved at: {contour_img_path}")
