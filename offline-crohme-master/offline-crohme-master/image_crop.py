import cv2
import numpy as np
import matplotlib.pyplot as plt


def crop_inside_box(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (adjust threshold value if needed)
    ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find external contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Look for the largest rectangular contour
    max_area = 0
    box_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > max_area:
            max_area = area
            box_contour = approx

    if box_contour is None:
        print("No box detected; returning original image.")
        return image

    # Get the bounding rectangle for the detected box
    x, y, w, h = cv2.boundingRect(box_contour)
    cropped = image[y:y + h, x:x + w]
    return cropped


if __name__ == "__main__":
    # Replace 'sample_input.jpg' with the path to your sample image
    image_path = "200923-131-2.inkml.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image at {image_path}")
    else:
        cropped_image = crop_inside_box(image)

        # Convert images from BGR to RGB for displaying with matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        # Display the original and cropped images side by side
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(cropped_rgb)
        plt.title("Cropped Image")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
