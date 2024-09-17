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
special_tokens = {"additional_special_tokens": ["###Human:", "###Assistant:"]}
TOKENIZER.add_special_tokens(special_tokens)


# Set the maximum sequence length
max_sequence_length = 1024
min_sequence_length = 256


# Data Processing Function
def process_item(item):
    input_text = (
        "###Human: Write a fairy tale about "
        + item["summary"]
        + "\n###Assistant: "
        + item["story"]
        + " ###"
    )
    try:
        # Tokenize the input_text to get its length
        tokenized_input = TOKENIZER(
            input_text,
            return_length=True,
            add_special_tokens=True,
        )

        total_length = tokenized_input["length"][0]
    except Exception as ex:
        print(f"Tokenization error: {ex}")
        return None

    if total_length > max_sequence_length:
        # print(
        #     f"Skipping item because total length {total_length} exceeds {max_sequence_length} tokens."
        # )
        return None  # Skip items that are too long
    elif total_length < min_sequence_length:
        # print(
        #     f"Skipping item because total length {total_length} is smaller than {min_sequence_length} tokens."
        # )
        return None  # Skip items that are too short
    else:
        return {
            "input_text": input_text,
            **TOKENIZER(
                input_text,
                truncation=True,
                max_length=max_sequence_length,
                padding="max_length",
            ),
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
TRAIN_DATA = [item for item in map(process_item, RAW_DATA) if item is not None]
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
    r=16, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

TOKENIZER.pad_token = TOKENIZER.eos_token

model.resize_token_embeddings(len(TOKENIZER))
training_args = transformers.TrainingArguments(
    per_device_train_batch_size=4,  # Use a batch size that fits your GPU memory
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    warmup_steps=5,
    max_steps=500,
    learning_rate=1e-5,
    fp16=True,
    logging_steps=25,
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
print("Training completed successfully.")

torch.cuda.empty_cache()

# Save model and tokenizer
trainer.save_model("adapter-model-new")
TOKENIZER.save_pretrained("adapter-model-new")  # Save the tokenizer with the model
print("Model and tokenizer saved successfully to 'adapter-model-new' directory.")

print(f"Total parameters in the model: {sum(p.numel() for p in model.parameters())}")
print(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)
