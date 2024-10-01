import json
import os


def is_valid_json_file(file_path):
    try:
        # Load the JSON data from the file
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        # Check if 'stories' key exists and has at least 10 stories
        if "stories" not in data or len(data["stories"]) < 10:
            return False

        # Collect user_ids to ensure they are unique
        user_ids = {story["user_id"] for story in data["stories"]}

        # Check if there are at least 15 unique user_ids
        return len(user_ids) >= 10

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading {file_path}: {e}")
        return False


def log_error(page_number, error_file="inkitt/error_log.txt"):
    # Append the page number to the error log file
    with open(error_file, "a", encoding="utf-8") as f:
        f.write(f"{page_number}\n")


# Clear the error log file before running
def clear_error_log(error_file="inkitt/error_log.txt"):
    with open(error_file, "w", encoding="utf-8") as f:
        f.write("")  # Just create/clear the file


#########
# # Create directory for error log if it doesn't exist
# os.makedirs("inkitt", exist_ok=True)


# clear_error_log()

# # Check each JSON file and log errors if necessary
# for number in range(1, 129):
#     file_path = f"inkitt/response_{number}.json"

#     if not os.path.exists(file_path):
#         log_error(number)
#         continue

#     if is_valid_json_file(file_path):
#         print(f"{file_path} is valid and contains at least 10 unique stories.")
#     else:
#         log_error(number)
#         print(f"{file_path} is invalid or does not contain enough stories.")
#########

#########
# # Load the first JSON file
# file_path = "inkitt/drama_list/response_1.json"

# # Attempt to load and print the first story
# try:
#     with open(file_path, "r", encoding="utf-8") as json_file:
#         data = json.load(json_file)

#     # Assuming the "stories" key exists and is a list, get the first story
#     if "stories" in data and len(data["stories"]) > 0:
#         first_story = data["stories"][0]
#         output_file_path = "inkitt/sample/sample_story.json"
#         with open(output_file_path, "w", encoding="utf-8") as outfile:
#             json.dump(first_story, outfile, indent=4)
#     else:
#         first_story = "No stories found in the JSON file."

# except (FileNotFoundError, json.JSONDecodeError) as e:
#     first_story = f"Error reading {file_path}: {e}"
#########

#########
import os
import json
import re


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def extract_story_info(directory_path, output_file_path):
    story_info = []

    # Get all JSON files and sort them
    json_files = [f for f in os.listdir(directory_path) if f.endswith(".json")]
    json_files.sort(key=natural_sort_key)

    # Iterate through sorted files
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                # Check if 'stories' key exists and is a list
                if "stories" in data and isinstance(data["stories"], list):
                    file_story_info = [
                        {
                            "id": story.get("id"),
                            "title": story.get("title"),
                            "category_one": story.get("category_one"),
                            "category_two": story.get("category_two"),
                            "chapters": story.get("published_chapters_count"),
                            "summary": story.get("summary"),
                        }
                        for story in data["stories"]
                        if story.get("language", "").lower() == "english"
                        and "id" in story
                        and "title" in story
                    ]
                    story_info.extend(file_story_info)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error processing {filename}: {e}")

    # Write the story info to the output file
    with open(output_file_path, "w", encoding="utf-8") as outfile:
        json.dump({"stories": story_info}, outfile, indent=4)

    print(
        f"Extracted info for {len(story_info)} English stories and saved to {output_file_path}"
    )


# extract_story_info("inkitt/drama_list", "inkitt/_metadata_drama.json")
#########


#########
def evaluate_chapter_counts(input_file):
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    if "stories" not in data or not isinstance(data["stories"], list):
        print("Invalid JSON format. 'stories' key is missing or not a list.")
        return

    # Initialize counters for chapter ranges
    one_chapter = 0
    two_to_five_chapters = 0
    six_to_fifteen_chapters = 0
    more_than_fifteen_chapters = 0

    for story in data["stories"]:
        chapters = story.get("chapters", 0)
        if chapters == 1:
            one_chapter += 1
        elif 2 <= chapters <= 5:
            two_to_five_chapters += 1
        elif 6 <= chapters <= 15:
            six_to_fifteen_chapters += 1
        elif chapters > 15:
            more_than_fifteen_chapters += 1

    # Output the results
    print(f"Stories with 1 chapter: {one_chapter}")
    print(f"Stories with 2 to 5 chapters: {two_to_five_chapters}")
    print(f"Stories with 6 to 15 chapters: {six_to_fifteen_chapters}")
    print(f"Stories with more than 15 chapters: {more_than_fifteen_chapters}")


# Example usage:
evaluate_chapter_counts("inkitt/_metadata_drama.json")
#########

#########
import json


def split_metadata_by_chapters(input_file, output_file_small, output_file_large):
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    if "stories" not in data or not isinstance(data["stories"], list):
        print("Invalid JSON format. 'stories' key is missing or not a list.")
        return

    # Separate stories based on chapter counts
    small_chapter_stories = []
    large_chapter_stories = []

    for story in data["stories"]:
        chapters = story.get("chapters", 0)
        if chapters <= 5:
            small_chapter_stories.append(story)
        else:
            large_chapter_stories.append(story)

    # Write stories with chapters count from 1 to 5 to the first file
    with open(output_file_small, "w", encoding="utf-8") as small_file:
        json.dump({"stories": small_chapter_stories}, small_file, indent=4)

    # Write stories with more than 5 chapters to the second file
    with open(output_file_large, "w", encoding="utf-8") as large_file:
        json.dump({"stories": large_chapter_stories}, large_file, indent=4)

    print(f"Stories with chapters 1-5 saved to {output_file_small}")
    print(f"Stories with chapters >5 saved to {output_file_large}")


# Example usage:
split_metadata_by_chapters(
    "inkitt/_metadata_drama.json",
    "inkitt/_metadata_drama_small.json",
    "inkitt/_metadata_drama_large.json",
)
#########
