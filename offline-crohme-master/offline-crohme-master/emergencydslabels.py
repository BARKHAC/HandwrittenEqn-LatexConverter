import os
import shutil

# Define paths
images_new_folder = "images_5k_new"
labels_folder = "my_newdataset/labels_processed_new"
new_labels_folder = "labels_5k_new"

# Create new folder if it doesn't exist
os.makedirs(new_labels_folder, exist_ok=True)

# Extract base names from images_new (removing .inkml.png)
image_names = {f.replace(".inkml.png", "") for f in os.listdir(images_new_folder) if f.endswith(".inkml.png")}

# Copy matching label files
for label_file in os.listdir(labels_folder):
    if label_file.rsplit(".", 1)[0] in image_names:  # Match without extension
        shutil.copy(os.path.join(labels_folder, label_file), os.path.join(new_labels_folder, label_file))

print(f"Copied {len(os.listdir(new_labels_folder))} label files to {new_labels_folder}")
