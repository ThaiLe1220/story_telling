import json
import transformers
import torch.cuda
import torch
import logging
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_scheduler,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

logger.info("Loading data from JSONL file...")
data = []
with open("dataset/grimm_stories_dataset.jsonl", "r", encoding="utf-8") as file:
    for line_number, line in enumerate(file, 1):
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_number}: {e}")
            print(f"Problematic line: {line}")

logger.info(f"Successfully loaded {len(data)} items from the JSONL file.")

logger.info("Initializing model and tokenizer...")
model_id = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

logger.info(f"Tokenizer initialized. Vocabulary size: {len(tokenizer)}")


def process_item(item):
    input_text = (
        "###Human: " + item["title"] + "###Assisstant: " + item["story"] + " ### "
    )
    return {
        "input_text": input_text,
        **tokenizer(input_text, truncation=True, max_length=512),
    }


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

logger.info("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

logger.info("Enabling gradient checkpointing...")
model.gradient_checkpointing_enable()
logger.info("Preparing model for kbit training...")
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
logger.info(f"LoRA config created: {config}")

data = list(map(process_item, data))
logger.info(f"Processed {len(data)} items.")

logger.info("Initializing trainer...")
tokenizer.pad_token = tokenizer.eos_token
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
    train_dataset=data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train()

torch.cuda.empty_cache()

logger.info("Starting training...")
try:
    trainer.train()
    logger.info("Training completed successfully.")
except Exception as e:
    logger.error(f"An error occurred during training: {str(e)}")

logger.info("Saving model...")
try:
    trainer.save_model("adapter-model")
    logger.info("Model saved successfully to 'adapter-model' directory.")
except Exception as e:
    logger.error(f"An error occurred while saving the model: {str(e)}")

logger.info("Evaluating model...")
try:
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")
except Exception as e:
    logger.error(f"An error occurred during evaluation: {str(e)}")

logger.info("Script execution completed.")

# Add some performance metrics
if torch.cuda.is_available():
    logger.info(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    logger.info(f"GPU Memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

logger.info(
    f"Total parameters in the model: {sum(p.numel() for p in model.parameters())}"
)
logger.info(
    f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)
