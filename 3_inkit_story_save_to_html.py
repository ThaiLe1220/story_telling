import json
import requests
import subprocess
import os


def download_chapter(story_id, chapter_num):
    url = f"https://www.inkitt.com/stories/drama/{story_id}/chapters/{chapter_num}"
    output_dir = f"inkitt/drama_stories/{story_id}"
    output_file = f"{output_dir}/chapter_{chapter_num}.html"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if the chapter file already exists and is not empty
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(
            f"Chapter {chapter_num} for story {story_id} already exists, skipping download."
        )
        return

    # Construct the curl command
    curl_command = [
        "curl",
        "-X",
        "GET",
        url,
        "-H",
        "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "-H",
        "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "-H",
        "Accept-Language: en-US,en;q=0.5",
        "-H",
        "Connection: keep-alive",
        "-H",
        "Upgrade-Insecure-Requests: 1",
        "-L",
        "-o",
        output_file,
    ]

    # Run the curl command
    try:
        result = subprocess.run(
            curl_command, check=True, capture_output=True, text=True
        )
        print(f"Successfully downloaded chapter {chapter_num}")

        # Check the size of the downloaded file
        file_size = os.path.getsize(output_file)
        print(f"Downloaded file size: {file_size} bytes")

    except subprocess.CalledProcessError as e:
        print(f"Failed to download chapter {chapter_num}")
        print(f"Error: {e}")
        print(f"Curl output: {e.output}")


def download_all_chapters(metadata_file_path):
    try:
        # Load the metadata file
        with open(metadata_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

            # Check if 'stories' key exists and is a list
            if "stories" in data and isinstance(data["stories"], list):
                for story in data["stories"]:
                    story_id = story.get("id")
                    chapter_count = story.get("chapters", 0)

                    # Ensure valid story_id and chapter count
                    if story_id and chapter_count > 0:
                        print(f"Downloading chapters for Story ID: {story_id}")

                        # Loop through each chapter and run download_chapter function
                        for chapter_num in range(1, chapter_count + 1):
                            try:
                                download_chapter(story_id, chapter_num)
                            except Exception as e:
                                print(
                                    f"Failed to download chapter {chapter_num} for story {story_id}: {e}"
                                )
                    else:
                        print(
                            f"Invalid story or chapter count for Story ID: {story_id}"
                        )
            else:
                print("No stories found in the metadata file.")

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error processing the metadata file: {e}")


# Example usage:
download_all_chapters("inkitt/_metadata_drama.json")
85198