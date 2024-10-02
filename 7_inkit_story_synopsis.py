import google.generativeai as genai
import os
import json
import ollama


# Set your API key as an environment variable
os.environ["GEMINI_API_KEY"] = "AIzaSyBY3laVu5u447QZqa1g99iXVNajll7kdmI"

# Configure the API with your key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


# Function to generate content with default settings
def generate_default_content(p):
    # Create a model instance for Gemini 1.5 Flash
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # Generate content based on the prompt
    response = model.generate_content(p)

    # Print the generated response
    print("Generated Response:")
    print(response.text)


# Function to generate content with custom settings
def generate_custom_content(p):
    # Define custom configuration for content generation
    model_config = {
        "temperature": 0.7,  # Controls randomness (0.0 - 1.0)
        "top_p": 0.9,  # Controls diversity via nucleus sampling
        "max_output_tokens": 256,  # Maximum number of tokens in the output
    }
    # Create a new model instance with custom configuration
    custom_model = genai.GenerativeModel(
        "gemini-1.5-flash", generation_config=model_config
    )

    # Generate content with custom settings
    custom_response = custom_model.generate_content(p)

    # Print the customized generated response
    print("\nCustomized Generated Response:")
    print(custom_response.text)


# Define the prompt for content generation
prompt = "Explain how AI works in simple terms."

#########
# Call the function for default content generation
# generate_default_content(prompt)

# Call the function for custom content generation
# generate_custom_content(prompt)
#########


def get_model_response(model_name, prompt):
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return response["message"]["content"]


def save_response_to_file(content, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)


try:
    # Load the metadata file
    with open("inkitt/_metadata_drama.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    # Check if 'stories' key exists and is a list
    if "stories" in data and isinstance(data["stories"], list):

        # Process and copy the first 20 stories
        for index, story in enumerate(data["stories"]):
            if index >= 1:  # Stop after copying 20 stories
                break

            story_id = story.get("id")
            story_title = story.get("title")
            story_category = story.get("category_one")
            story_chapters = story.get("chapters")
            story_summary = story.get("summary").replace("\n", " ")

            file_path = f"inkitt/test_merge/{story_id}.txt"

            with open(file_path, "r", encoding="utf-8") as file:
                full_story_text = file.read()

            if full_story_text and story_id:
                questionToAsk = f"""
Instruction: Summarize the given story as a concise synopsis following this structure. Use the metadata for context but base the synopsis solely on the Input text.

Story Metadata (for context only):
Title: {story_title}
Category: {story_category}
Total Chapters: {story_chapters}
Original Summary: {story_summary}

Synopsis Structure:
1. Hook (1-2 sentences): Capture the essence of the story with an intriguing opening.
2. Setting and Protagonists (2-3 sentences): Introduce the main characters and establish the time and place.
3. Inciting Incident (1-2 sentences): Describe the catalyst event that sets the story in motion.
4. Central Conflict (1 sentence): Clearly state the main problem or goal driving the narrative.
5. Key Plot Points (3-4 sentences): Highlight major events and turning points in the story's progression.
6. Climax (1-2 sentences): Describe the peak moment of tension or conflict resolution.
7. Resolution (1-2 sentences): Explain how the story concludes and any character growth.
8. Themes (1 sentence): Identify 2-3 central themes explored in the story.

Additional Guidelines:
- Aim for a total length of 200-250 words.
- Present the synopsis as a cohesive narrative without section headers or numbers.
- Use present tense and active voice throughout.
- Maintain the story's tone (e.g., suspenseful, humorous, dramatic).
- Include only essential character names and details.
- For series or complex stories, focus on the core narrative arc.

Input text:
{full_story_text}
                """

                # print(story_id, story_title, story_category, story_summary)
                # print(questionToAsk)

                # Get responses from both models
                OllamaResponse_31 = get_model_response("llama3.1:latest", questionToAsk)
                OllamaResponse_32 = get_model_response("llama3.2:latest", questionToAsk)

                # Save responses to files
                save_response_to_file(OllamaResponse_31, "OutputOllama3.1_8b.txt")
                save_response_to_file(OllamaResponse_32, "OutputOllama3.2_3b.txt")

                print("Responses have been generated and saved to files.")
    else:
        print("No stories found in the metadata file.")

except (json.JSONDecodeError, IOError) as e:
    print(f"Error processing the metadata file: {e}")
