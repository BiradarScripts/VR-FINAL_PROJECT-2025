import os
import logging
import csv
import json
from dotenv import load_dotenv
import google.generativeai as genai
from google.ai.generativelanguage import Content, Part, Blob
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
import traceback
import time

# Load environment variables
load_dotenv()

# Retrieve API key from environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise Exception("API key not found. Please set the GOOGLE_API_KEY environment variable.")
else:
    genai.configure(api_key=API_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize the Google Gemini model
llm = genai.GenerativeModel('gemini-2.0-flash')

# Template for image analysis
generic_template_freshness_detection = ''' analyze image '''
# Structured prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", generic_template_freshness_detection),
    ("user", "{text}")
])

# Output parser
parser = StrOutputParser()

# Load metadata from CSV
def load_image_metadata(csv_path):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            images = [row for row in reader]
        return images
    except Exception as e:
        logger.error(f"Error reading metadata file: {e}")
        return []

# Clean text function
def clean_text(text):
    return text.strip().strip('"').strip('\\')

def save_analysis_result(image_id, analysis_data, output_dir="analyzed_images"):
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Parse the raw data
        raw_data = json.dumps(analysis_data)  # Convert analysis_data to a string representation
        blocks = raw_data.split('\\n\\n')

        qa_list = []
        for block in blocks:
            lines = block.strip().split('\\n')
            if len(lines) == 2:
                try:
                    question_line = lines[0].strip()
                    answer_line = lines[1].strip()

                    # Use partition to safely split at the first colon
                    _, _, question_value = question_line.partition(':')
                    _, _, answer_value = answer_line.partition(':')

                    qa_list.append({
                        "question": clean_text(question_value),
                        "answer": clean_text(answer_value)
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Skipping block due to error: {block} — {e}")

        # Save the parsed list to a new JSON file
        parsed_file = os.path.join(output_dir, f"{image_id}.json")
        with open(parsed_file, 'w') as outfile:
            json.dump(qa_list, outfile, indent=2)

        logger.info(f"✅ Parsed {len(qa_list)} Q&A pairs to: {parsed_file}")
    
    except Exception as e:
        logger.error(f"Error parsing and saving result for {image_id}: {e}")

# Check and resume state
def check_and_resume_state(state_file="process_state.json"):
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                last_processed_image = state.get("last_processed_image")
                logger.info(f"Resuming from image_id {last_processed_image}...")
                return last_processed_image
        else:
            logger.info("No state file found, starting fresh.")
            return None
    except Exception as e:
        logger.error(f"Error checking the state: {e}")
        return None

# Update state
def update_state(image_id, state_file="process_state.json"):
    try:
        state = {"last_processed_image": image_id}
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=4)
        logger.info(f"Updated process state with image_id {image_id}")
    except Exception as e:
        logger.error(f"Error updating state file: {e}")

# Load local image
def load_local_image(image_path):
    try:
        with Image.open(image_path) as img:
            img_byte_array = img.tobytes()
        return img_byte_array
    except Exception as e:
        logger.error(f"Error loading image from {image_path}: {e}")
        return None

# Analyze image
def analyze_image(image_path: str):
    try:
        # Read the image file as bytes
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        # Determine the MIME type based on file extension
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.jpg') or image_path.lower().endswith('.jpeg'):
            mime_type = 'image/jpeg'
        else:
            raise ValueError("Unsupported image format. Only PNG and JPEG are supported.")

        # Prepare content parts with image data and instructions
        content_parts = [
            Part(text=generic_template_freshness_detection),  # System message with instructions
            Part(inline_data=Blob(mime_type=mime_type, data=image_data))  # The image as binary data
        ]

        # Generate the content (stream=True for real-time generation)
        response = llm.generate_content(Content(parts=content_parts), stream=True)
        response.resolve()

        # Parse the AI's analysis result
        parsed_response = parser.invoke(response.text)

        # Return the parsed result
        return parsed_response

    except Exception as e:
        if "429" in str(e):
            logger.error(f"Quota exceeded: {e}")
            raise Exception("Quota exceeded, stopping process.")
        print(f"An error occurred: {e}")
        return str(e)

# Process images
def process_images(csv_path="abo-images-small/images/metadata/images.csv", output_dir="analyzed_images"):
    images = load_image_metadata(csv_path)
    last_processed_image = check_and_resume_state()

    for image in images:
        image_id = image["image_id"]
        image_path = os.path.join("abo-images-small", "images", "small", image["path"])
        print(image_path)
        if last_processed_image and image_id <= last_processed_image:
            continue

        image_data = load_local_image(image_path)
        if image_data:
            try:
                analysis_result = analyze_image(image_path)
                save_analysis_result(image_id, analysis_result, output_dir)
                update_state(image_id)
            except Exception as e:
                if "Quota exceeded" in str(e):
                    # If quota is exceeded, stop the process and save the last processed image.
                    update_state(image_id)  # Save the last processed image before stopping
                    logger.error("Process stopped due to quota exceed error.")
                    break

if __name__ == "__main__":
    process_images(csv_path="abo-images-small/images/metadata/images.csv")
