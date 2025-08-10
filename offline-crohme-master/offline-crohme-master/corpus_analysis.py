import re
import statistics
from collections import Counter

# Define a list of known LaTeX commands to capture as whole tokens.
special_tokens = [
    r"\\frac", r"\\sin", r"\\cos", r"\\sqrt", r"\\sum",
    r"\\forall", r"\\exists", r"\\log", r"\\pi", r"\\int",
    r"\\alpha", r"\\beta", r"\\gamma", r"\\leq", r"\\geq", r"\\infty"
]
special_pattern = "|".join(re.escape(token) for token in special_tokens)

# Regex pattern to tokenize a LaTeX expression.
# It captures:
#  - Special tokens
#  - Dollar-enclosed expressions
#  - Sequences of non-whitespace characters excluding certain punctuation
#  - Individual symbols such as braces, underscores, carets, backslashes, and dollars.
pattern = rf"({special_pattern}|\$[^\$]+\$|[^\s{{}}_^\\$]+|[{{}}_^\\$])"


def clean_line(line):
    """
    Clean a single line from the corpus:
    - Strip extra whitespace.
    - Skip lines that are too short or don't contain any LaTeX commands.
    - Filter out obvious noise (e.g., lines containing "Toto").
    Returns None if the line should be skipped.
    """
    line = line.strip()
    if len(line) < 5 or "\\" not in line:
        return None
    if "Toto" in line:
        return None
    return line


def tokenize_line(line):
    """
    Tokenize the input line using the defined regex pattern.
    """
    tokens = re.findall(pattern, line)
    return tokens


def analyze_tokens(lines, label=""):
    """
    Analyze a list of lines:
    - Tokenize each line.
    - Count token frequencies.
    - Print average frequency, most common tokens, and tokens at minimum frequency.
    """
    token_counter = Counter()
    for line in lines:
        tokens = tokenize_line(line)
        token_counter.update(tokens)

    frequencies = list(token_counter.values())
    average_frequency = statistics.mean(frequencies) if frequencies else 0

    print(f"Analysis for {label}:")
    print(f"  Total unique tokens: {len(token_counter)}")
    print(f"  Average token frequency: {average_frequency:.2f}")
    print("  Most common tokens:")
    for token, freq in token_counter.most_common(10):
        print(f"    {token}: {freq}")

    if frequencies:
        min_freq = min(frequencies)
        least_common = [(token, freq) for token, freq in token_counter.items() if freq == min_freq]
    else:
        least_common = []

    print("  Tokens occurring at minimum frequency:")
    for token, freq in least_common:
        print(f"    {token}: {freq}")
    print("\n")
    return token_counter


def main():
    original_file = "latex_corpus.txt"
    cleaned_file = "cleaned_latex_corpus.txt"

    # Read the original corpus file.
    with open(original_file, "r", encoding="utf-8") as f:
        original_lines = f.readlines()

    # Analyze tokens from the original corpus.
    original_lines_stripped = [line.strip() for line in original_lines if line.strip()]
    print(f"Original Corpus: {len(original_lines_stripped)} non-empty lines found.\n")
    analyze_tokens(original_lines_stripped, label="Original Corpus")

    # Clean the corpus lines.
    cleaned_lines = []
    for line in original_lines:
        cleaned = clean_line(line)
        if cleaned:
            cleaned_lines.append(cleaned)

    print(f"Cleaned Corpus: {len(cleaned_lines)} lines remain after cleaning.\n")

    # Write cleaned lines to a new file.
    with open(cleaned_file, "w", encoding="utf-8") as out_f:
        for line in cleaned_lines:
            out_f.write(line + "\n")

    # Analyze tokens from the cleaned corpus.
    analyze_tokens(cleaned_lines, label="Cleaned Corpus")


if __name__ == "__main__":
    main()
