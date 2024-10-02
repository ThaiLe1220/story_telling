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
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

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
            if index >= 10:  # Stop after copying 20 stories
                break

            story_id = story.get("id")
            story_title = story.get("title")
            story_category = story.get("category_one")
            story_chapters = story.get("chapters")
            story_summary = story.get("summary").replace("\n", " ")

            if story_id:
                for i in range(story_chapters):
                    file_path = f"inkitt/test/{story_id}/chapter_{i+1}.txt"
                    with open(file_path, "r", encoding="utf-8") as file:
                        full_story_text = file.read()

                    questionToAsk = f"""
Story Title: {story_title}, Chapter: {i+1}, Category: {story_category}
Story Summary: {story_summary}

Instruction: Create a concise synopsis of the given story, strictly adhering to the following structure.

Synopsis Structure:
- Hook: Capture the essence of the story with an intriguing opening.
- Setting and Protagonists: Introduce the main characters and establish the time and place.
- Inciting Incident: Describe the catalyst event that sets the story in motion.
- Central Conflict: Clearly state the main problem or goal driving the narrative.
- Key Plot Points(focus on this): Highlight major events and turning points in the story's progression.
- Climax: Describe the peak moment of tension or conflict resolution.
- Resolution: Explain how the story concludes and any character growth.
- Themes: List 2-3 central themes explored in the story.

Guidelines:
- Be concise and focused, Use Active Language, Avoid Excessive Detail
- Total length around 200 words 
- Only output the synopsis structure, Dont output anything else

Input text:
{full_story_text}
                    """
                    print(story_id, i + 1)

                    # Get responses from both models
                    OllamaResponse = get_model_response(
                        "llama3.1:latest", questionToAsk
                    )

                    # Save responses to files
                    save_response_to_file(
                        OllamaResponse, f"inkitt/test_syn/{story_id}/{i+1}.txt"
                    )

    else:
        print("No stories found in the metadata file.")

except (json.JSONDecodeError, IOError) as e:
    print(f"Error processing the metadata file: {e}")
