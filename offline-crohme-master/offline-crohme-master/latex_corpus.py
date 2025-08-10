import os

labels_dir = "labels_processed_new/"  # Change this to your actual labels directory
output_file = "Latex_Corpus_new.txt"

# Gather all LaTeX expressions
with open(output_file, "w", encoding="utf-8") as out_file:
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(labels_dir, filename), "r", encoding="utf-8") as f:
                latex_expression = f.read().strip()
                out_file.write(latex_expression + "\n")

print(f"LaTeX corpus saved to {output_file}")
