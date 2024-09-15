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
min_sequence_length = 256


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


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_HF_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)


TOKENIZER.pad_token = TOKENIZER.eos_token
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=6,  # was 10
    gradient_accumulation_steps=4,  # was 4
    warmup_steps=2,
    max_steps=500,
    learning_rate=1e-4,
    fp16=False,
    logging_steps=1,
    output_dir="outputs",
    optim="paged_adamw_8bit",
)

torch.cuda.empty_cache()
trainer = transformers.Trainer(
    model=model,
    train_dataset=TRAIN_DATA,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(TOKENIZER, mlm=False),
)
model.config.use_cache = False
trainer.train()

torch.cuda.empty_cache()

try:
    trainer.train()
    print("Training completed successfully.")
except Exception as e:
    print(f"An error occurred during training: {str(e)}")

try:
    trainer.save_model("adapter-model")
    print("Model saved successfully to 'adapter-model' directory.")
except Exception as e:
    print(f"An error occurred while saving the model: {str(e)}")


print(f"Total parameters in the model: {sum(p.numel() for p in model.parameters())}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)
