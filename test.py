import json
import transformers
import torch.cuda
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Model and Tokenizer Setup
MODEL_HF_NAME = "meta-llama/Meta-Llama-3.1-8B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_HF_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token

# Add special tokens to the tokenizer
special_tokens = ["### Instruction:", "### Response:"]
TOKENIZER.add_tokens(special_tokens)
print(f"Added special tokens: {special_tokens}")

# Set the maximum sequence length
max_sequence_length = 4069
min_sequence_length = 512


# Data Processing Function
def process_item(item):
    description = item.get("summary", "").strip()
    story = item.get("story", "").strip()

    if not description or not story:
        print(f"Missing 'description' or 'story' in item: {item}")
        return None  # Skip items with missing data

    # Create the input text using the new format
    input_text = (
        "### Instruction:\n"
        f"Write a fairy tale about {description.lower()}\n\n"
        "### Response:\n"
        f"{story}\n"
    )

    # Tokenize the input_text to get its length
    tokenized_input = TOKENIZER(
        input_text,
        return_length=True,
        add_special_tokens=True,
    )

    total_length = tokenized_input["length"][0]

    if total_length > max_sequence_length:
        print(
            f"Skipping item because total length {total_length} exceeds {max_sequence_length} tokens."
        )
        return None  # Skip items that are too long
    elif total_length < min_sequence_length:
        print(
            f"Skipping item because total length {total_length} is smaller than {min_sequence_length} tokens."
        )
        return None  # Skip items that are too short

    print(
        f"Processing item with description '{description}' of total length {total_length} tokens."
    )

    # Tokenize again with truncation and padding
    tokenized_input = TOKENIZER(
        input_text,
        truncation=True,
        max_length=max_sequence_length,
        padding="max_length",
        add_special_tokens=True,
    )

    return {
        "input_text": input_text,
        "input_ids": tokenized_input["input_ids"],
        "attention_mask": tokenized_input["attention_mask"],
    }


# Load the data from the JSONL file
RAW_DATA = []
with open("dataset/summarized_stories_cleaned.jsonl", "r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, 1):
        line = line.strip()
        if not line:
            continue
        try:
            RAW_DATA.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_number}: {e}")
            print(f"Problematic line: {line}")

print(f"[Workflow] Successfully loaded {len(RAW_DATA)} items from the JSONL file.")

# Process the data
TRAIN_DATA = []
for item in RAW_DATA:
    processed_item = process_item(item)
    if processed_item:
        TRAIN_DATA.append(processed_item)

print(f"[Workflow] Successfully processed {len(TRAIN_DATA)} items for training.")

# # Optional: Print out a sample of the processed data for verification
# if TRAIN_DATA:
#     sample_item = TRAIN_DATA[0]
#     # print("Sample processed item:")
#     # print(f"Input Text:\n{sample_item['input_text']}")
#     print(f"Input IDs (first 50 tokens):\n{sample_item['input_ids'][:50]}...")
#     # print(f"Attention Mask (first 50 tokens):\n{sample_item['attention_mask'][:50]}...")
