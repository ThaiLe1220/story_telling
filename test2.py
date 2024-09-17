from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig

# Load the base model
base_model_name = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained("adapter-model")
tokenizer.pad_token = tokenizer.eos_token

# Verify special tokens are present
special_tokens = {"additional_special_tokens": ["###Human:", "###Assistant:"]}
if not all(
    token in tokenizer.additional_special_tokens
    for token in special_tokens["additional_special_tokens"]
):
    tokenizer.add_special_tokens(special_tokens)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the fine-tuned model with LoRA adapters
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)

# Ensure that the tokenizer's pad token is set
tokenizer.pad_token = tokenizer.eos_token

# Resize the model embeddings
model.resize_token_embeddings(len(tokenizer))

# Load the LoRA adapters
model = PeftModel.from_pretrained(model, "adapter-model")
model.eval()


test_inputs = [
    "###Human: Write a fairy tale about a brave little toaster ###Assistant: ",
    "###Human: Write a fairy tale about a dragon who lost his fire ###Assistant: ",
    "###Human: Write a fairy tale about the talking fish, the kind fisherman, and the sea Witch's Curse ###Assistant: ",
]

encoded_inputs = tokenizer(
    test_inputs,
    return_tensors="pt",
    padding=True,
).to(model.device)

# Set generation parameters
generation_args = {
    "max_length": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "eos_token_id": tokenizer.eos_token_id,
}

# Generate responses
with torch.no_grad():
    generated_outputs = model.generate(
        input_ids=encoded_inputs["input_ids"],
        attention_mask=encoded_inputs["attention_mask"],
        **generation_args,
    )

# Decode the outputs
responses = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)

# Print the responses
for i, response in enumerate(responses):
    print(f"Input: {test_inputs[i]}")
    print(f"Generated Response: {response.split('###Assistant:')[-1].strip()}")
    print("-" * 50)
