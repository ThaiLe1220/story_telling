# import json

# # Load JSON data from file (assuming you saved the response as 'response.json')
# with open("response.json", "r") as file:
#     data = json.load(file)

# # Extract stories from the JSON response
# stories = data.get("stories", [])

# # Analyze stories
# for story in stories:
#     # Extract relevant information
#     title = story.get("title", "No title").strip()
#     author = story.get("user", {}).get("name", "Unknown author")
#     language = story.get("language", "Unknown language")
#     age_rating = story.get("age_rating", "Not rated")
#     overall_rating = story.get("overall_rating_cache", "No rating")
#     word_count = story.get("words_count", "Unknown word count")
#     chapters = story.get("chapters_count", "Unknown chapter count")

#     # Print analysis of each story
#     print(f"Title: {title}")
#     print(f"Author: {author}")
#     print(f"Language: {language}")
#     print(f"Age Rating: {age_rating}")
#     print(f"Overall Rating: {overall_rating}")
#     print(f"Word Count: {word_count}")
#     print(f"Chapters: {chapters}")
#     print("-" * 40)

# # Additional analysis, e.g., average rating or total word count, can be done here if needed.
# total_word_count = sum(story.get("words_count", 0) for story in stories)
# print(f"Total word count across all stories: {total_word_count}")


import subprocess
import os

# Create directory if it doesn't exist
os.makedirs("inkitt", exist_ok=True)


def get_error_pages(error_file="inkitt/error_log.txt"):
    """Read the error log file and return a set of page numbers."""
    if os.path.exists(error_file):
        with open(error_file, "r", encoding="utf-8") as f:
            return {line.strip() for line in f}
    return set()


# Get the list of pages from the error log
error_pages = get_error_pages()

# # Loop from 1 to 129
# for number in range(100, 129):
#     # Skip if number is not in the error log
#     if str(number) not in error_pages:
#         print(f"Skipping page {number} as it is not in the error log.")
#         continue

#     # Define the curl command with the current number
#     curl_command = [
#         "curl",
#         "-X",
#         "GET",
#         f"https://www.inkitt.com/genres/drama/{number}?exclude_age_ratings=&exclude_story_lengths=&exclude_story_statuses=work_in_progress%2Cexcerpt&exclude_sub_genres=erotica&locale=en&period=alltime&sort=popular",
#         "-H",
#         "Accept: application/json",
#         "-H",
#         "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
#         "-o",
#         f"inkitt/response_{number}.json",
#     ]

#     # Execute the curl command
#     try:
#         subprocess.run(curl_command, check=True)
#         print(f"Response saved to inkitt/response_{number}.json")
#     except subprocess.CalledProcessError as e:
#         print(f"An error occurred for page {number}: {e}")


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


def log_error(page_number, error_file="inkitt/drama_list/error_log.txt"):
    # Append the page number to the error log file
    with open(error_file, "a", encoding="utf-8") as f:
        f.write(f"{page_number}\n")


# Clear the error log file before running
def clear_error_log(error_file="inkitt/drama_list/error_log.txt"):
    with open(error_file, "w", encoding="utf-8") as f:
        f.write("")  # Just create/clear the file


# Create directory for error log if it doesn't exist
os.makedirs("inkitt", exist_ok=True)

clear_error_log()

# Check each JSON file and log errors if necessary
for number in range(1, 129):
    file_path = f"inkitt/drama_list/response_{number}.json"

    if not os.path.exists(file_path):
        log_error(number)
        continue

    if is_valid_json_file(file_path):
        print(f"{file_path} is valid and contains at least 10 unique stories.")
    else:
        log_error(number)
        print(f"{file_path} is invalid or does not contain enough stories.")
