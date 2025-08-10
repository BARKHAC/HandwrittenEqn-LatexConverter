# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Optionally import pytesseract if available for orientation detection
# try:
#     import pytesseract
#
#     USE_TESSERACT = True
# except ImportError:
#     print("pytesseract not installed; using fallback method.")
#     USE_TESSERACT = False
#
#
# def correct_orientation(image):
#     """
#     Corrects the orientation of the input image.
#     First tries using pytesseract's OSD to determine the angle.
#     If that fails, uses a fallback method based on image contours.
#     """
#     if USE_TESSERACT:
#         try:
#             osd_output = pytesseract.image_to_osd(image)
#             # Example OSD output line: "Orientation in degrees: 270"
#             angle = 0
#             for line in osd_output.split("\n"):
#                 if "Orientation in degrees" in line:
#                     angle = int(line.split(":")[1].strip())
#                     break
#             # pytesseract returns the angle the image is rotated clockwise.
#             # To correct it, we rotate counterclockwise by that angle.
#             if angle != 0:
#                 (h, w) = image.shape[:2]
#                 center = (w // 2, h // 2)
#                 M = cv2.getRotationMatrix2D(center, -angle, 1.0)
#                 rotated = cv2.warpAffine(image, M, (w, h),
#                                          flags=cv2.INTER_CUBIC,
#                                          borderMode=cv2.BORDER_REPLICATE)
#                 return rotated
#             else:
#                 return image
#         except Exception as e:
#             print("Pytesseract orientation detection failed:", e)
#
#     # Fallback: Use a contour-based approach
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Invert the image so that text is white on a black background
#     gray_inv = cv2.bitwise_not(gray)
#     # Threshold the image
#     thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     # Find coordinates of all non-zero pixels
#     coords = np.column_stack(np.where(thresh > 0))
#     # Get the minimum area rectangle that contains the text
#     angle = cv2.minAreaRect(coords)[-1]
#     # Adjust the angle
#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h),
#                              flags=cv2.INTER_CUBIC,
#                              borderMode=cv2.BORDER_REPLICATE)
#     return rotated
#
#
# if __name__ == "__main__":
#     # Replace 'sample_input.jpg' with the path to your sample image
#     input_image_path = "images/200923-131-249.inkml.png"
#     image = cv2.imread(input_image_path)
#     if image is None:
#         print("Error loading image. Check the file path.")
#     else:
#         corrected_image = correct_orientation(image)
#
#         # Convert images from BGR to RGB for displaying with matplotlib
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         corrected_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
#
#         # Display the original and corrected images side by side
#         plt.figure(figsize=(12, 6))
#         plt.subplot(1, 2, 1)
#         plt.imshow(image_rgb)
#         plt.title("Original Image")
#         plt.axis("off")
#         plt.subplot(1, 2, 2)
#         plt.imshow(corrected_rgb)
#         plt.title("Corrected Image")
#         plt.axis("off")
#         plt.show()


# import cv2
# import matplotlib.pyplot as plt
#
#
# def flip_image_sequence(image_path):
#     # Read the original image
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Could not read the image at {image_path}")
#         return
#
#     # Flip left-to-right (horizontal flip)
#     flipped_lr = cv2.flip(image, 1)
#
#     # Flip the horizontally flipped image upside-down (vertical flip)
#     flipped_ud = cv2.flip(flipped_lr, 0)
#
#     # Now flip the final output (flipped_ud) from right-to-left (another horizontal flip)
#     final_output = cv2.flip(flipped_ud, 1)
#
#     # Convert images from BGR to RGB for displaying with matplotlib
#     original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     flipped_lr_rgb = cv2.cvtColor(flipped_lr, cv2.COLOR_BGR2RGB)
#     flipped_ud_rgb = cv2.cvtColor(flipped_ud, cv2.COLOR_BGR2RGB)
#     final_output_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
#
#     # Display all images side by side
#     plt.figure(figsize=(16, 6))
#
#     plt.subplot(1, 4, 1)
#     plt.imshow(original_rgb)
#     plt.title("Original")
#     plt.axis("off")
#
#     plt.subplot(1, 4, 2)
#     plt.imshow(flipped_lr_rgb)
#     plt.title("Flipped L-R")
#     plt.axis("off")
#
#     plt.subplot(1, 4, 3)
#     plt.imshow(flipped_ud_rgb)
#     plt.title("Flipped Upside-Down")
#     plt.axis("off")
#
#     plt.subplot(1, 4, 4)
#     plt.imshow(final_output_rgb)
#     plt.title("Final: Flipped R-L")
#     plt.axis("off")
#
#     plt.tight_layout()
#     plt.show()
#
#
# if __name__ == "__main__":
#     image_path = "images/200923-131-60.inkml.png"  # Replace with your image file path
#     flip_image_sequence(image_path)
#
#
import cv2
import glob
import os

def process_image(image_path):
    # Read the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at {image_path}")
        return None

    # Flip left-to-right (horizontal flip)
    flipped_lr = cv2.flip(image, 1)

    # Flip the horizontally flipped image upside-down (vertical flip)
    flipped_ud = cv2.flip(flipped_lr, 0)

    # Now flip the final output (flipped_ud) from right-to-left (another horizontal flip)
    final_output = cv2.flip(flipped_ud, 1)

    return final_output

def process_all_images(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # Get all image files in the input folder (adjust pattern if needed, e.g., *.png)
    image_paths = glob.glob(os.path.join(input_folder, "*"))
    print(f"Found {len(image_paths)} images in {input_folder}")

    for image_path in image_paths:
        final_img = process_image(image_path)
        if final_img is not None:
            file_name = os.path.basename(image_path)
            output_path = os.path.join(output_folder, file_name)
            cv2.imwrite(output_path, final_img)
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Skipping {image_path} due to read error.")

if __name__ == "__main__":
    input_folder = "images"          # Folder containing the original images
    output_folder = "images_oriented"  # Folder to save the processed images
    process_all_images(input_folder, output_folder)
