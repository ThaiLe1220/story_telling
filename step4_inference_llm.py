import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the merged model and tokenizer once
merged_model_path = "grimm_story_model"
tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set

model = AutoModelForCausalLM.from_pretrained(
    merged_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Match with bnb_4bit_compute_dtype
)
model.config.use_cache = True  # Enable cache for faster inference


def generate_fairy_tale_merged(story, max_new_tokens=1024):
    try:
        # Prepare the input text
        text = f"###Human: Write a fairy tale about {story}\n###Assistant:"

        # Tokenize the input text
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # Generate text using the model
        sample_outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,  # Pass the attention mask
            pad_token_id=tokenizer.eos_token_id,  # Use tokenizer's eos_token_id
            do_sample=True,
            max_new_tokens=max_new_tokens,  # Generate up to max_new_tokens tokens
            top_p=0.95,
            top_k=45,
            temperature=0.7,
            no_repeat_ngram_size=2,
            repetition_penalty=1,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode and return the generated text
        output_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        print(output_text)
        # Post-process to extract the assistant's response
        assistant_start = text + " "
        if assistant_start in output_text:
            output_text = output_text.split(assistant_start)[1]
        else:
            output_text = output_text[len(text) :]

        # Stop at the next "###" if present
        output_text = output_text.split("###")[0].strip()

        return output_text

    except Exception as e:
        print(f"An error occurred during generation: {e}")
        return None


# Example usage
story = "a girl who finds a mirror that reveals people's hidden intentions"
merged_output = generate_fairy_tale_merged(story, max_new_tokens=1024)

# a clever shoemaker who makes a deal with a mischievous forest spirit
# a forgotten princess trapped in a tower of ice
# a brave woodcutter who befriends a golden bear
# a magic soup pot that never runs empty
# three sisters who set out to break a curse on their village


# print("Merged Model Output:")
# print(merged_output)

# a poor little boy
# a clever shoemaker who makes a deal with a mischievous forest spirit


# MODEL_HF_NAME = "meta-llama/Meta-Llama-3.1-8B"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# llama_tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_NAME)
# llama_model = AutoModelForCausalLM.from_pretrained(
#     MODEL_HF_NAME,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True,
# )


# def generate_fairy_tale_original(story, max_new_tokens=1024):

#     input_ids = llama_tokenizer.encode(story, return_tensors="pt").to("cuda:0")

#     sample_outputs = llama_model.generate(
#         input_ids,
#         pad_token_id=50256,
#         do_sample=True,
#         max_new_tokens=512,
#         top_p=0.90,
#         top_k=50,
#         temperature=0.7,
#         early_stopping=False,
#         no_repeat_ngram_size=2,
#         num_return_sequences=1,
#     )

#     return llama_tokenizer.decode(sample_outputs[0].tolist())


# original_output = generate_fairy_tale_original(
#     "Write a fairy tale about three sisters who set out to break a curse on their village",
#     max_new_tokens=1024,
# )
# print("Original Model Output:")
# print(original_output)
