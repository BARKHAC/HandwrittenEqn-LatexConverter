import os
import shutil

# Define source and destination directories
source_dir = "data_processed/"  # Change this to your actual directory
images_dir = "images/"
labels_dir = "labels/"

# Create destination directories if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Get all .png and .txt file names (without extensions)
png_files = {f[:-4] for f in os.listdir(source_dir) if f.endswith(".png")}
txt_files = {f[:-4] for f in os.listdir(source_dir) if f.endswith(".txt")}

# Find missing files
missing_png = txt_files - png_files
missing_txt = png_files - txt_files

# Move files to respective folders
for file in png_files:
    shutil.move(os.path.join(source_dir, file + ".png"), os.path.join(images_dir, file + ".png"))
    if file in txt_files:
        shutil.move(os.path.join(source_dir, file + ".txt"), os.path.join(labels_dir, file + ".txt"))

# Logging missing files
with open("missing_files.log", "w") as log:
    if missing_png:
        log.write("Missing PNG files for the following labels:\n" + "\n".join(missing_png) + "\n")
    if missing_txt:
        log.write("Missing TXT files for the following images:\n" + "\n".join(missing_txt) + "\n")

print("Separation complete! Check missing_files.log for any missing pairs.")
