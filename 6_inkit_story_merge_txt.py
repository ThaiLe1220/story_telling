import os


def merge_chapters_into_one(directory, output_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate through all subdirectories (story IDs) in the directory
    for story_id in os.listdir(directory):
        story_path = os.path.join(directory, story_id)

        if os.path.isdir(story_path):
            # Initialize the content to store all chapters for the current story
            merged_content = ""

            # Get a list of all chapter files in the story directory
            chapter_files = sorted(
                [f for f in os.listdir(story_path) if f.endswith(".txt")]
            )

            # Read and merge each chapter file
            for chapter_file in chapter_files:
                chapter_path = os.path.join(story_path, chapter_file)

                # Read the content of the chapter
                with open(chapter_path, "r", encoding="utf-8") as file:
                    chapter_content = file.read().strip()

                # Append the chapter content and delimiter
                merged_content += chapter_content + "\n\n###\n\n"

            # Define the output file path for the merged content
            output_file_path = os.path.join(output_directory, f"{story_id}.txt")

            # Write the merged content to the new file
            with open(output_file_path, "w", encoding="utf-8") as output_file:
                output_file.write(merged_content)

            print(f"Story {story_id} chapters merged into {output_file_path}")


def count_characters_and_words(directory):
    # Print header for the output
    print(f"{'Story ID':<18}{'Characters':<18}{'Words':<15}")
    print("-" * 50)

    # Iterate over all files in the directory
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        # Ensure we're processing only .txt files
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

                # Count characters and words
                character_count = len(content)
                word_count = len(content.split())

            # Print the results in a consistent format
            print(f"{file_name:<18}{character_count:<18}{word_count:<15}")


source_directory = "inkitt/test"
output_directory = "inkitt/test_merge"

#########
# merge_chapters_into_one(source_directory, output_directory)
#########
count_characters_and_words(output_directory)
#########
