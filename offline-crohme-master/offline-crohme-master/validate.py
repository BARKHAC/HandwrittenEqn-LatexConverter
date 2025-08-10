import os

labels_dir = "labels_5k_new/"
images_dir = "images_5k_new/"
log_file = "missing_files.log"

# Get list of all .txt files in labels directory
label_files = {f.replace(".txt", "") for f in os.listdir(labels_dir) if f.endswith(".txt")}

# Get list of all .png files in images directory
image_files = {f.replace(".inkml.png", "") for f in os.listdir(images_dir) if f.endswith(".png")}

# Find mismatches
missing_images = label_files - image_files
missing_labels = image_files - label_files

# Log missing files
with open(log_file, "w") as log:
    if missing_images:
        log.write("Missing images for the following labels:\n")
        for file in missing_images:
            log.write(f"{file}.txt\n")

    if missing_labels:
        log.write("\nMissing labels for the following images:\n")
        for file in missing_labels:
            log.write(f"{file}.inkml.png\n")

print(f"Validation complete. Log saved in {log_file}")
