import os
import json
import shutil


#########
def analyze_story(story_id, directory):
    # Ensure story_id is a string before joining with path
    story_path = os.path.join(directory, str(story_id))
    if os.path.isdir(story_path):
        chapter_files = [f for f in os.listdir(story_path) if f.endswith(".txt")]
        chapter_count = len(chapter_files)

        single_chapter_count = 1 if chapter_count == 1 else 0
        multi_chapter_count = 1 if chapter_count > 1 else 0
        copyright_chapters = []
        untitled_chapters = []

        for chapter_file in chapter_files:
            chapter_path = os.path.join(story_path, chapter_file)
            chapter_number = chapter_file.split(".")[0].split("_")[-1]

            with open(chapter_path, "r", encoding="utf-8") as file:
                first_line = file.readline().strip()
                if first_line.startswith("Title: Copyright"):
                    copyright_chapters.append((story_id, chapter_number))
                elif first_line.startswith("Title: Untitled chapter"):
                    untitled_chapters.append((story_id, chapter_number))

        return (
            single_chapter_count,
            multi_chapter_count,
            copyright_chapters,
            untitled_chapters,
        )
    return 0, 0, [], []


def process_all_stories(metadata_file_path, directory):
    total_stories = 0
    single_chapter = 0
    multi_chapter = 0
    copyright_chapters = []
    untitled_chapters = []

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
                    # print(f"Processing Story Index {index}: Story ID: {story_id}")

                    # Call the analyze_story function
                    sc, mc, cc, uc = analyze_story(story_id, directory)

                    single_chapter += sc
                    multi_chapter += mc
                    copyright_chapters.extend(cc)
                    untitled_chapters.extend(uc)

                    total_stories += 1
                else:
                    print(f"Invalid story or chapter count for Story ID: {story_id}")
        else:
            print("No stories found in the metadata file.")

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error processing the metadata file: {e}")

    # Print results
    print("Story Statistics:")
    print(f"Total number of stories: {total_stories}")
    print(f"Stories with only 1 chapter: {single_chapter}")
    print(f"Stories with more than 1 chapter: {multi_chapter}")

    print("\nChapter Analysis:")
    print("Chapters starting with 'Title: Copyright':")
    for story_id, chapter_number in copyright_chapters:
        print(f"Story ID: {story_id}, Chapter: {chapter_number}")

    print("\nChapters starting with 'Title: Untitled chapter':")
    for story_id, chapter_number in untitled_chapters:
        print(f"Story ID: {story_id}, Chapter: {chapter_number}")

    print(f"\nTotal 'Copyright' chapters: {len(copyright_chapters)}")
    print(f"Total 'Untitled' chapters: {len(untitled_chapters)}")


# Replace this with the actual path to your metadata and story directory
metadata_file_path = "inkitt/_metadata_drama.json"
directory = "inkitt/drama_stories_txt"

process_all_stories(metadata_file_path, directory)


#########


def copy_first_20_stories(metadata_file_path, source_directory, destination_directory):
    try:
        # Load the metadata file
        with open(metadata_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Check if 'stories' key exists and is a list
        if "stories" in data and isinstance(data["stories"], list):
            # Ensure the destination directory exists
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)

            # Process and copy the first 20 stories
            for index, story in enumerate(data["stories"]):
                if index >= 20:  # Stop after copying 20 stories
                    break

                story_id = story.get("id")
                if story_id:
                    source_story_path = os.path.join(source_directory, str(story_id))
                    destination_story_path = os.path.join(
                        destination_directory, str(story_id)
                    )

                    if os.path.isdir(source_story_path):
                        # Copy the story directory to the destination
                        shutil.copytree(source_story_path, destination_story_path)
                        print(f"Copied Story ID {story_id} to {destination_story_path}")
                    else:
                        print(f"Story directory not found for Story ID {story_id}")
        else:
            print("No stories found in the metadata file.")

    except (json.JSONDecodeError, IOError) as e:
        print(f"Error processing the metadata file: {e}")


# Usage example
metadata_file_path = "inkitt/_metadata_drama.json"
source_directory = "inkitt/drama_stories_txt"
destination_directory = "inkitt/test"

copy_first_20_stories(metadata_file_path, source_directory, destination_directory)
