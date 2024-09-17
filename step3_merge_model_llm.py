# Import necessary libraries
from transformers import (
    AutoTokenizer,  # For tokenizing input text
    AutoModelForCausalLM,  # For loading the language model
    BitsAndBytesConfig,  # For configuring quantization
)
from peft import PeftModel  # For loading fine-tuned adapters
import torch  # PyTorch library for tensor computations

# Specify the base model ID
MODEL_HF_NAME = "meta-llama/Meta-Llama-3.1-8B"

# Load the base model and tokenizer
BASE_MODEL = AutoModelForCausalLM.from_pretrained(
    MODEL_HF_NAME,
    return_dict=True,  # Return a dictionary of outputs
    torch_dtype=torch.float16,  # Use half-precision floating point
    trust_remote_code=True,  # Trust code from the model's repo
)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_HF_NAME)
TOKENIZER.pad_token = TOKENIZER.eos_token
special_tokens = {"additional_special_tokens": ["###Human:", "###Assistant:"]}
TOKENIZER.add_special_tokens(special_tokens)
BASE_MODEL.resize_token_embeddings(len(TOKENIZER))

# Load the fine-tuned adapter and apply it to the base model
model = PeftModel.from_pretrained(BASE_MODEL, "adapter-model").to("cuda")

# Merge the adapter weights with the base model
model = model.merge_and_unload()

# Save the merged model and TOKENIZER
model.save_pretrained("grimm_story_model")
TOKENIZER.save_pretrained("grimm_story_model")

print("Model merged and saved successfully!")
