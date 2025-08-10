# import os
# import glob
#
# # Define input and output directories
# input_dir = "labels"
# output_dir = "labels_processed_new"
#
# # Create the output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#     print(f"Created directory: {output_dir}")
#
# # Process each .txt file in the input directory
# txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
# print(f"Found {len(txt_files)} files in {input_dir}")
#
# for file_path in txt_files:
#     with open(file_path, "r", encoding="utf-8") as f:
#         content = f.read().strip()  # Remove leading/trailing whitespace
#
#     # If the content starts and ends with a dollar sign, remove them
#     if content.startswith("$") and content.endswith("$") and len(content) >= 2:
#         cleaned_content = content[1:-1].strip()  # also strip again in case of spaces after removal
#     else:
#         cleaned_content = content
#
#     # Write the cleaned content to a new file in the output directory
#     file_name = os.path.basename(file_path)
#     output_path = os.path.join(output_dir, file_name)
#     with open(output_path, "w", encoding="utf-8") as f_out:
#         f_out.write(cleaned_content)
#
#     print(f"Processed {file_name} -> {output_path}")


