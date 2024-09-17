import random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def load_model_and_tokenizer(model_path):
    double_quant_nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=double_quant_nf4_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def generate_story(model, tokenizer, prompt, max_length=4096):
    input_ids = tokenizer.encode(prompt.lower(), return_tensors="pt").to("cuda:0")
    sample_outputs = model.generate(
        input_ids,
        pad_token_id=50256,
        do_sample=True,
        max_length=max_length,
        top_p=0.95,
        top_k=45,
        temperature=0.5,
        early_stopping=False,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        repetition_penalty=1.2,
    )
    generated_text = tokenizer.decode(sample_outputs[0].tolist())
    return generated_text


def save_stories_to_file(stories, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for summary, story in stories:
            f.write(f"Summary: {summary}\n\n")
            f.write(f"{story}\n\n")
            f.write("-" * 50 + "\n\n")
    print(f"All stories saved to {filename}")


def main():
    merged_model = "grimm_story_model"
    model, tokenizer = load_model_and_tokenizer(merged_model)

    summaries = [
        "The Poor Boy's Magical Journey Through the Enchanted Forest",
        # "The Brave Girl's Quest to Save Her Village from the Witch",
        # "The Clever Fox's Trick to Outsmart the Greedy Wolf King",
        # "The Magical Wishing Tree and the Orphan's Three Wishes",
        # "The Lost Princess's Adventure with the Talking Animals of Whispering Woods",
        # "The Mysterious Old Woman's Secret Garden of Forgotten Dreams",
        # "The Greedy King's Lesson: A Tale of Gold and Kindness",
        # "The Talking Fish, the Kind Fisherman, and the Sea Witch's Curse",
        # "The Enchanted Forest's Guardian: A Tale of Nature's Magic",
        # "The Cursed Prince, the Kind Maiden, and the Spell of Truth",
    ]

    generated_stories = []

    for summary in summaries:
        print(f"Generating story for: {summary}")
        prompt = f"Write a fairy tale about {summary.lower()}"
        generated_story = generate_story(model, tokenizer, prompt)
        generated_stories.append((summary, generated_story))
        print(f"Story generated for: {summary}")

    filename = "grimm_style_fairy_tales_collection.txt"
    save_stories_to_file(generated_stories, filename)


if __name__ == "__main__":
    main()
