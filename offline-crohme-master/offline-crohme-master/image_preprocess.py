import os
import cv2
import numpy as np

# Paths
IMAGE_DIR = "images/"  # Your raw images folder
OUTPUT_DIR = "images_preprocessed/"  # Folder to save preprocessed images

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image size for TrOCR (ViT expects 384x384)
IMAGE_SIZE = (384, 384)

def preprocess_image(image_path, output_path):
    """Resizes and normalizes an image for TrOCR."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
    img = cv2.resize(img, IMAGE_SIZE)  # Resize to 384x384
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]

    # Save processed image
    cv2.imwrite(output_path, (img * 255).astype(np.uint8))  # Convert back to uint8 for saving

if __name__ == "__main__":
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
    for img_name in images:
        input_path = os.path.join(IMAGE_DIR, img_name)
        output_path = os.path.join(OUTPUT_DIR, img_name)
        preprocess_image(input_path, output_path)

    print(f"✅ Processed {len(images)} images and saved to {OUTPUT_DIR}")
