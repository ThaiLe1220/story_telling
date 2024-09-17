# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Configure 4-bit quantization
double_quant_nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load the merged model
merged_model = "grimm_story_model"
model = AutoModelForCausalLM.from_pretrained(
    merged_model,
    quantization_config=double_quant_nf4_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load the tokenizer for the merged model
TOKENIZER = AutoTokenizer.from_pretrained(merged_model)

# Prepare the input text
text = "###Human: Write a fairy tale about about a poor little boy ###Assistant: "

# Tokenize the input text and move it to GPU
input_ids = TOKENIZER.encode(text, return_tensors="pt").to("cuda:0")

# Generate text using the model
sample_outputs = model.generate(
    input_ids,
    pad_token_id=50256,
    do_sample=True,
    max_length=2048,
    top_p=0.95,
    top_k=45,
    temperature=0.4,
    early_stopping=False,
    no_repeat_ngram_size=4,
    num_return_sequences=1,
    repetition_penalty=1.2,
)

# Print the generated text
for i, sample_output in enumerate(sample_outputs):
    print(
        ">> Generated text {}\n\n{}".format(
            i + 1, TOKENIZER.decode(sample_output.tolist())
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

# MODEL_HF_NAME = "meta-llama/Meta-Llama-3.1-8B"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# TOKENIZER = AutoTokenizer.from_pretrained(MODEL_HF_NAME)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_HF_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
# )
# text = "Generate a Grim-style fairy tale about a poor little boy"
# input_ids = TOKENIZER.encode(text, return_tensors="pt").to("cuda:0")
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
#             i + 1, TOKENIZER.decode(sample_output.tolist())
#         )
#     )
#     print("\n---")
