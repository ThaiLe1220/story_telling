import random
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
import torch
import re
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load model and tokenizer
try:
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, return_dict=True, torch_dtype=torch.float16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, "adapter-model").to("cuda")
    logger.info("Running merge_and_unload")
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Configure 4-bit quantization
    double_quant_nf4_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
    )

    # Load the quantized model
    model_id = "merged_model"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=double_quant_nf4_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise


def generate_story(title):
    prompt = f"###Human: {title}###Assisstant: "
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(
        input_ids,
        pad_token_id=50256,
        do_sample=True,
        max_new_tokens=1024,  # Adjusted to match training max_length
        top_p=0.9,
        top_k=50,
        temperature=0.8,
        early_stopping=False,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
    )
    return tokenizer.decode(output[0].tolist(), skip_special_tokens=True)


# List of random story titles
titles = [
    "The Poor Little Boy's Adventure",
    "The Brave Young Girl and the Witch",
    "The Clever Fox's Trick",
    "The Magical Tree of Wishes",
    "The Lost Princess and the Talking Animals",
    "The Mysterious Old Woman's Secret",
    "The Greedy King's Lesson",
    "The Talking Fish and the Fisherman",
    "The Enchanted Forest's Guardian",
    "The Cursed Prince and the Kind Maiden",
]

# Generate 10 stories and save them
with open("grimm_stories.txt", "w", encoding="utf-8") as file:
    for i in range(10):
        title = random.choice(titles)
        story = generate_story(title)

        # Extract content (remove the prompt)
        content = story.split("###Assisstant: ", 1)[-1].strip()

        # Write to file
        file.write(f"Story {i+1}: {title}\n\n")
        file.write(f"{content}\n\n")
        file.write("=" * 50 + "\n\n")

        print(f"Generated story {i+1}: {title}")

print("All stories have been generated and saved to 'grimm_stories.txt'")
