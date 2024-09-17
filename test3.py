import json
import transformers
import torch.cuda
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


# Model and Tokenizer Setup
MODEL_HF_NAME = "meta-llama/Meta-Llama-3.1-8B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_HF_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token

# Add special tokens to the tokenizer
special_tokens = {"additional_special_tokens": ["###Human:", "###Assistant:"]}
TOKENIZER.add_special_tokens(special_tokens)


# Set the maximum sequence length
max_sequence_length = 2048


# Data Processing Function
def process_item(item):
    input_text = (
        "###Human: Write a fairy tale about " + item["summary"] + "\n###Assistant:"
    )
    target_text = item["story"] + " ###"

    # Tokenize input and target separately
    input_ids = TOKENIZER(
        input_text,
        truncation=True,
        max_length=max_sequence_length,
        padding="max_length",
        return_tensors="pt",
    )["input_ids"].squeeze()

    labels = TOKENIZER(
        target_text,
        truncation=True,
        max_length=max_sequence_length,
        padding=False,
        return_tensors="pt",
    )["input_ids"].squeeze()

    # Check if labels length is within the desired range
    if len(labels) < 512 or len(labels) > 2048:
        return None  # Skip this item

    # Create attention mask
    attention_mask = (input_ids != TOKENIZER.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
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

# Convert RAW_DATA to a Hugging Face Dataset
dataset = Dataset.from_list(RAW_DATA)

# Split the dataset into training and validation sets
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Process the datasets
train_dataset = train_dataset.map(process_item)
eval_dataset = eval_dataset.map(process_item)

print(f"[Workflow] Successfully processed {len(train_dataset)} training items.")
print(f"[Workflow] Successfully processed {len(eval_dataset)} evaluation items.")

# Bits and Bytes Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# Load the model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_HF_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Enable gradient checkpointing and prepare model
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# LoRA Config
peft_config = LoraConfig(
    r=16,  # Increased rank for better learning capacity
    lora_alpha=32,
    lora_dropout=0.1,  # Increased dropout to prevent overfitting
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# Resize embeddings
TOKENIZER.pad_token = TOKENIZER.eos_token
model.resize_token_embeddings(len(TOKENIZER))

# Training Arguments
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Adjusted for effective batch size
    warmup_steps=100,
    num_train_epochs=4,  # Training over multiple epochs
    learning_rate=5e-5,  # Reduced learning rate
    fp16=True,
    logging_steps=50,
    output_dir="outputs",
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=100,  # More frequent evaluation
    save_steps=100,
    load_best_model_at_end=True,
    report_to="none",  # Disable reporting to third-party services
)

# Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=TOKENIZER,
    padding=True,
    max_length=max_sequence_length,
    return_tensors="pt",
)

# Trainer
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Disable caching (important for training)
model.config.use_cache = False

# Start Training
trainer.train()

print("Training completed successfully.")

torch.cuda.empty_cache()

# Save model and tokenizer
trainer.save_model("adapter-model-new")
TOKENIZER.save_pretrained("adapter-model-new")
print("Model and tokenizer saved successfully to 'adapter-model-new' directory.")

print(f"Total parameters in the model: {sum(p.numel() for p in model.parameters())}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)
