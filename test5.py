import cv2
import numpy as np
import os


def process_and_save_steps(file_path):
    # Create an 'outputs' directory if it doesn't exist
    output_dir = './outputs3'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    # Extract the base name of the file for later use in naming the output images
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # Read the image in grayscale
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Step names for saving files
    steps = ['OriginalImage', 'BlurredImage', 'BinaryImage', 'Opening', 'Contours']

    # Apply a Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use Otsu's thresholding method to binarize the processed image
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations to remove small noise and to separate the knife edge from the surface
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find the contours in the image
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour is the knife surface and filter out the rest
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        contours = [max_contour]

    # Draw contours on the original image
    contour_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for display
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

    # Save each step image with the specified naming convention
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_{steps[0]}.png'), image)
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_{steps[1]}.png'), blurred)
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_{steps[2]}.png'), binary)
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_{steps[3]}.png'), opening)
    cv2.imwrite(os.path.join(output_dir, f'{base_name}_{steps[4]}.png'), contour_img)

    # Return the paths of the saved images
    saved_files = [os.path.join(output_dir, f'{base_name}_{step}.png') for step in steps]
    return saved_files


folder_path = "./imgs2"
file_names = os.listdir(folder_path)
for file_name in file_names:
    process_and_save_steps(f'./imgs2/{file_name}')
# process_and_save_steps(f'./imgs/5.jpg')

