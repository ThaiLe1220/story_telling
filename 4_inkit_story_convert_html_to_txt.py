import os
import json
from bs4 import BeautifulSoup


def extract_chapter_info(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract the title
    title = soup.find("h2", class_="chapter-head-title")
    title_text = title.text.strip() if title else "No title found"

    # Extract the content
    content_div = soup.find("div", class_="story-page-text")

    if content_div:
        # Extract all paragraphs
        paragraphs = content_div.find_all("p")
        content = "\n\n".join([p.text for p in paragraphs])
    else:
        content = "No content found"

    return title_text, content


def process_story_chapters(story_id, chapter_count):
    input_dir = f"inkitt/drama_stories/{story_id}"
    output_dir = f"inkitt/drama_stories_txt/{story_id}"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for chapter_num in range(1, chapter_count + 1):
        input_file = f"{input_dir}/chapter_{chapter_num}.html"
        output_file = f"{output_dir}/chapter_{chapter_num}.txt"

        if os.path.exists(input_file):
            # Read HTML content
            with open(input_file, "r", encoding="utf-8") as file:
                html_content = file.read()

            # Extract title and content
            title, content = extract_chapter_info(html_content)

            # Save title and content to a text file
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(f"Title: {title}\n\n")
                file.write(content)

            print(f"Processed Story {story_id}, Chapter {chapter_num}")
        else:
            print(f"HTML file not found for Story {story_id}, Chapter {chapter_num}")


def process_all_stories(metadata_file_path):
    try:
        # Load the metadata file
        with open(metadata_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Check if 'stories' key exists and is a list
        if "stories" in data and isinstance(data["stories"], list):
            # Limit processing to the first 100 stories
            for index, story in enumerate(data["stories"]):
                if index >= 100:  # Stop after processing 100 stories
                    break

                story_id = story.get("id")
                chapter_count = story.get("chapters", 0)

                # Ensure valid story_id and chapter count
                if story_id and chapter_count > 0:
                    print(f"Processing Story ID: {story_id}")
                    process_story_chapters(story_id, chapter_count)
                else:
                    print(f"Invalid story or chapter count for Story ID: {story_id}")
        else:
            print("No stories found in the metadata file.")

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error processing the metadata file: {e}")


process_all_stories("inkitt/_metadata_drama.json")
