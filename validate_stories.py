import os
import re
import logging
from transformers import AutoTokenizer
from tabulate import tabulate
from statistics import mean

# import nltk

# nltk.download("punkt")
# from nltk.tokenize import sent_tokenize


# Logging setup
logging.basicConfig(
    filename="stories/analysis_log.txt",
    level=logging.INFO,
    # format="%(asctime)s - %(message)s",
    format="%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Model and Tokenizer Setup
MODEL_HF_NAME = "meta-llama/Meta-Llama-3.1-8B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_HF_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token

# Add special tokens to the tokenizer
special_tokens = {"additional_special_tokens": ["###Human:", "###Assistant:"]}
TOKENIZER.add_special_tokens(special_tokens)


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def get_token_count(text):
    return len(TOKENIZER.encode(text))


def get_char_count(text):
    return len(text)


def analyze_directory(directory):
    logging.info("-" * 50)
    logging.info(f"Analyzing directory: {directory}")

    try:
        txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
        txt_files.sort()

        if not txt_files:
            logging.info(f"No .txt files found in the '{directory}' directory.")
            return

        data = []
        for file_name in txt_files:
            file_path = os.path.join(directory, file_name)
            content = read_file(file_path)
            token_count = get_token_count(content)
            char_count = get_char_count(content)
            data.append([file_name, token_count, char_count])

        # # Sort data by token count in descending order
        # data.sort(key=lambda x: x[1], reverse=True)

        # Calculate summary statistics
        token_counts = [row[1] for row in data]
        char_counts = [row[2] for row in data]

        summary = [
            ["Minimum", min(token_counts), min(char_counts)],
            ["Maximum", max(token_counts), max(char_counts)],
            ["Average", f"{mean(token_counts):.2f}", f"{mean(char_counts):.2f}"],
            ["Total", sum(token_counts), sum(char_counts)],
        ]

        # Combine file data and summary into one table
        headers = ["File/Statistic", "Tokens", "Characters"]
        combined_data = data + [["---", "---", "---"]] + summary

        # Create and log the table
        table = tabulate(combined_data, headers=headers, tablefmt="grid")
        logging.info(f"Analysis for {directory}:\n{table}\n")

    except Exception as e:
        logging.error(f"An error occurred while analyzing {directory}: {str(e)}")


if __name__ == "__main__":
    filename = "stories-note.txt"
    try:
        with open(filename, "r", encoding="utf-8") as file:
            count = 0
            for line in file:
                stripped_line = line.strip()
                if stripped_line:
                    sentences = stripped_line.split(".")
                    first_sentence = sentences[0] if sentences else ""

                    count += 1
                    logging.info(f"Prompt {count}: {first_sentence}")

    except FileNotFoundError:
        logging.error(f"Error: File '{filename}' not found.")
    except IOError:
        logging.error(f"Error: Unable to read file '{filename}'.")

    logging.info("")

    directories = [
        "stories/ChatGPT4o",
        "stories/ChatGPTo1-preview",
        "stories/Claude3.5Sonnet",
        "stories/PerplexitySonarHuge",
    ]

    for directory in directories:
        analyze_directory(directory)

    logging.info("Analysis complete. Check 'analysis_log.txt' for full results.")
    print("Analysis complete. Check 'analysis_log.txt' for full results.")
