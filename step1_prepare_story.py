import os
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import concurrent.futures
import requests

from datasets import load_dataset
from groq import Groq
import threading
import random
import time


def grimm_scraper():
    # URL of the page to scrape
    url = "https://www.cs.cmu.edu/~spok/grimmtmp/"

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all list items containing links to stories
        stories = soup.find_all("li")

        # Create a directory to save the text files
        output_dir = "grimm_stories"
        os.makedirs(output_dir, exist_ok=True)

        # Function to download a story
        def download_story(story):
            link = story.find("a")
            if link:
                title = link.text
                href = link["href"]
                story_url = url + href

                # Send a GET request to download the story content
                story_response = requests.get(story_url)

                if story_response.status_code == 200:
                    # Create a safe filename by replacing invalid characters
                    safe_title = (
                        title.replace("'", "").replace(" ", "_").replace("/", "_")
                    )
                    file_path = os.path.join(output_dir, f"{safe_title}.txt")

                    # Write the content to the text file
                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(story_response.text)
                    return f"Saved: {file_path}"
                else:
                    return f"Failed to retrieve {title}. Status code: {story_response.status_code}"

        # Use multithreading to download stories with tqdm progress bar
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(download_story, story) for story in stories]
            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                print(future.result())
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")


def download_thanhnew_grim():
    # Load the dataset
    dataset = load_dataset("thanhnew2001/grim")

    # Access the data (assuming it's in the "train" split)
    data = dataset["train"]

    # Open a file to write the JSONL data
    with open("dataset/grim_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            # Create a dictionary for each item
            json_item = {"title": item["title"], "story": item["story"]}
            # Write the JSON object as a string, followed by a newline
            f.write(json.dumps(json_item, ensure_ascii=False) + "\n")

    print("Dataset saved as 'dataset/grim_dataset.jsonl'")


def convert_raw_stories_to_jsonl():
    # Directory containing the text files
    input_dir = "grimm_stories"
    # Output JSONL file
    output_file = "dataset/grimm_stories_dataset.jsonl"

    # Ensure the input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    # List to store all story entries
    stories = []

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):  # Assuming all story files are .txt
            file_path = os.path.join(input_dir, filename)

            # Extract title from filename (remove .txt extension)
            title = os.path.splitext(filename)[0]

            # Read the story content
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()

            # Create a dictionary for this story
            story_entry = {"title": title, "story": content}

            stories.append(story_entry)

    # Write all stories to the JSONL file
    with open(output_file, "w", encoding="utf-8") as jsonl_file:
        for story in stories:
            json.dump(story, jsonl_file, ensure_ascii=False)
            jsonl_file.write("\n")

    print(
        f"Conversion complete. {len(stories)} stories have been saved to '{output_file}'."
    )


# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Thread-local storage for client instances
thread_local = threading.local()


def rate_limited_api_call(client, messages, model, temperature, max_tokens):
    # Implement rate limiting logic here if needed
    return client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return thread_local.client


def summarize_story(filename):
    input_dir = "grimm_stories"
    output_dir = "summarized_grimm_stories"
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    prompt = f"Please summarize these Grimms' fairy tales story in 300 words:\n\n{content}\n\nOnly output the story content, dont output anything else"

    models = [
        "llama3-70b-8192",
        "llama3-70b-versatile",
        "llama3-8b-instant",
        "llama-3.1-8b-8192",
    ]
    client = get_client()

    for model_name in models:
        attempt = 0
        while attempt < 2:  # Try each model up to 2 times
            try:
                response = rate_limited_api_call(
                    client,
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                    temperature=0.5,
                    max_tokens=1024,
                )
                summary = response.choices[0].message.content

                with open(output_path, "w", encoding="utf-8") as file:
                    file.write(summary)

                print(f"Successfully summarized {filename} using {model_name}")
                return filename, "summarized"

            except Exception as e:
                if "rate_limit_exceeded" in str(e):
                    wait_time = exponential_backoff(attempt)
                    print(
                        f"Rate limit hit for {filename} with {model_name}. Retrying in {wait_time:.2f} seconds."
                    )
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    print(f"Error processing {filename} with {model_name}: {str(e)}")
                    break  # Move to the next model if it's not a rate limit error

        print(
            f"Failed to summarize {filename} with {model_name} after {attempt} attempts."
        )

    print(f"Failed to summarize {filename} with all available models.")
    return filename, "failed"


def exponential_backoff(attempt):
    return min(300, (2**attempt) + random.uniform(0, 1))


def summarize_grimm_stories():
    input_dir = "grimm_stories"
    output_dir = "summarized_grimm_stories"

    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files_to_process = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(summarize_story, filename) for filename in files_to_process
        ]

        summarized_count = 0
        failed_count = 0

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(files_to_process),
            desc="Processing stories",
        ):
            filename, status = future.result()
            if status == "summarized":
                summarized_count += 1
            elif status == "failed":
                failed_count += 1

            # Add a small delay between submissions
            time.sleep(0.5)

    print("Processing complete:")
    print(f"- {summarized_count} stories were summarized.")
    print(f"- {failed_count} stories failed to summarize.")
    print(f"All successful summaries are saved in '{output_dir}'.")
    if failed_count > 0:
        print(
            f"Warning: {failed_count} stories failed to summarize. You may want to retry these manually."
        )


def merge_jsonl_files(input_files, output_file):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for input_file in input_files:
            if not os.path.exists(input_file):
                print(f"Warning: File {input_file} does not exist. Skipping.")
                continue

            print(f"Processing {input_file}...")
            with open(input_file, "r", encoding="utf-8") as infile:
                for line in infile:
                    try:
                        json.loads(line)
                        outfile.write(line)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON in {input_file}: {line.strip()}")

    print(f"Merged {len(input_files)} files into {output_file}")


if __name__ == "__main__":
    print()

    ## Download preset data from hf and save in jsonl format
    # download_thanhnew_grim()

    ## Scrape and save grimm stories 1 file per story
    # grimm_scraper()

    ## Summarize all grimm stories using llama3.1-70b-8192
    # summarize_grimm_stories()

    ## Convert stories file to jsonl format
    # convert_raw_stories_to_jsonl()

    # Merge multiple jsonl files into one
    input_files = [
        "/Users/lehongthai/story_telling/dataset/grim_dataset.jsonl",
        "/Users/lehongthai/story_telling/dataset/grimm_stories_dataset.jsonl",
    ]

    # Run the merge function
    merge_jsonl_files(
        input_files, "/Users/lehongthai/story_telling/dataset/merged_output.jsonl"
    )
