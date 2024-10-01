import os


def analyze_stories_and_chapters(directory):
    story_count = 0
    single_chapter_count = 0
    multi_chapter_count = 0
    copyright_chapters = []
    untitled_chapters = []

    for story_id in os.listdir(directory):
        story_path = os.path.join(directory, story_id)
        if os.path.isdir(story_path):
            story_count += 1
            chapter_files = [f for f in os.listdir(story_path) if f.endswith(".txt")]
            chapter_count = len(chapter_files)

            if chapter_count == 1:
                single_chapter_count += 1
            else:
                multi_chapter_count += 1

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
        story_count,
        single_chapter_count,
        multi_chapter_count,
        copyright_chapters,
        untitled_chapters,
    )


# Replace this with the actual path to your drama_stories_txt directory
directory = "/Users/lehongthai/story_telling/inkitt/drama_stories_txt"

total_stories, single_chapter, multi_chapter, copyright_chapters, untitled_chapters = (
    analyze_stories_and_chapters(directory)
)

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
