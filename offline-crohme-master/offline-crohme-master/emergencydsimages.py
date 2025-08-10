# import os
# import shutil
# import random
#
# # Define paths
# images_folder = "my_newdataset/images"
# new_images_folder = "images_enew"
# num_images_to_copy = 5000  # Change to 3000 if needed
#
# # Create the new images folder if it doesn't exist
# os.makedirs(new_images_folder, exist_ok=True)
#
# # Get all image files (assuming they end with .png)
# image_files = [f for f in os.listdir(images_folder) if f.endswith(".png")]
#
# # Randomly select 5K images (or use [:5000] for first 5K)
# selected_images = random.sample(image_files, min(num_images_to_copy, len(image_files)))
#
# # Copy selected images
# for img in selected_images:
#     shutil.copy(os.path.join(images_folder, img), os.path.join(new_images_folder, img))
#
# print(f"Copied {len(os.listdir(new_images_folder))} images to {new_images_folder}")

import os
import random
import shutil

# Define paths
main_folder = "my_newdataset/images"       # Change this to your main folder path
subset_folder = "dataset_5k/images_5k"       # Change this to your subset folder path
new_folder = "images_5k_new"     # Change this to your new folder path

# Ensure the new folder exists
os.makedirs(new_folder, exist_ok=True)

# Get image lists
main_images = set(os.listdir(main_folder))
subset_images = set(os.listdir(subset_folder))

# Get new images that are not in the subset
remaining_images = list(main_images - subset_images)

# Select 5K random new images
new_selection = random.sample(remaining_images, 5000)

# Move selected images to the new folder
for img in new_selection:
    src_path = os.path.join(main_folder, img)
    dest_path = os.path.join(new_folder, img)
    shutil.move(src_path, dest_path)

print("5K new images moved successfully to images_5k_new.")

