# Import necessary libraries
from transformers import (
    AutoTokenizer,  # For tokenizing input text
    AutoModelForCausalLM,  # For loading the language model
    BitsAndBytesConfig,  # For configuring quantization
)
from peft import PeftModel  # For loading fine-tuned adapters
import torch  # PyTorch library for tensor computations

# Specify the base model ID
model_id = "meta-llama/Meta-Llama-3.1-8B"

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    return_dict=True,  # Return a dictionary of outputs
    torch_dtype=torch.float16,  # Use half-precision floating point
    trust_remote_code=True,  # Trust code from the model's repo
)

# Load the fine-tuned adapter and apply it to the base model
model = PeftModel.from_pretrained(base_model, "adapter-model").to("cuda")

print(f"Running merge_and_unload")
# Merge the adapter weights with the base model
model = model.merge_and_unload()

# Load the tokenizer for the base model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save the merged model and tokenizer
model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")

# Configure 4-bit quantization
double_quant_nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Use 'normal float' quantization
    bnb_4bit_use_double_quant=True,  # Use double quantization for further memory savings
)

# Update model_id to point to the merged model
model_id = "merged_model"

# Load the merged model with 4-bit quantization
model_double = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=double_quant_nf4_config,
    device_map="auto",  # Automatically determine device mapping
    trust_remote_code=True,
)

# Load the tokenizer for the merged model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prepare the input text
text = (
    "###Human: Generate a Grim-style fairy tale about a poor little boy###Assistant: "
)

# Tokenize the input text and move it to GPU
input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda:0")

# Generate text using the model
sample_outputs = model.generate(
    input_ids,
    pad_token_id=50256,  # Padding token ID
    do_sample=True,  # Use sampling for generation
    max_length=1024,  # Maximum length of generated text
    top_p=90,  # Nucleus sampling parameter
    top_k=50,  # Top-k sampling parameter
    temperature=0.8,  # Temperature for controlling randomness
    early_stopping=False,  # Don't stop early
    no_repeat_ngram_size=2,  # Avoid repeating 2-grams
    num_return_sequences=1,  # Number of sequences to generate
)

# Print the generated text
for i, sample_output in enumerate(sample_outputs):
    print(
        ">> Generated text {}\n\n{}".format(
            i + 1, tokenizer.decode(sample_output.tolist())
        )
    )
    print("\n---")

# import torch
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM,
#     AutoModelForCausalLM,
#     BitsAndBytesConfig,
# )

# model_id = "meta-llama/Meta-Llama-3.1-8B"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
# )
# text = "Generate a Grim-style fairy tale about a poor little boy"
# input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda:0")
# sample_outputs = model.generate(
#     input_ids,
#     pad_token_id=50256,
#     do_sample=True,
#     max_length=1000,
#     top_p=90,
#     top_k=50,
#     temperature=0.8,
#     early_stopping=False,
#     no_repeat_ngram_size=2,
#     num_return_sequences=1,
# )

# for i, sample_output in enumerate(sample_outputs):
#     print(
#         ">> Generated text {}\n\n{}".format(
#             i + 1, tokenizer.decode(sample_output.tolist())
#         )
#     )
#     print("\n---")
