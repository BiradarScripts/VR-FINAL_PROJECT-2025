import os
import logging
import csv
import json
import subprocess
from dotenv import load_dotenv, set_key
import google.generativeai as genai
from google.ai.generativelanguage import Content, Part, Blob
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
import traceback
import time
from google.api_core.exceptions import ResourceExhausted
import sys

# Load environment variables
load_dotenv()

API_KEY_FILE = "api_key.txt"
ENV_FILE = ".env"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load and set the first API key from api_key.txt
def load_and_set_api_key():
    with open(API_KEY_FILE, 'r') as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise Exception("No API keys found in api_key.txt")

    current_key = keys[0]
    set_key(ENV_FILE, "GOOGLE_API_KEY", current_key)
    genai.configure(api_key=current_key)
    return keys

# Rotate API key after exceeding retry attempts
def rotate_api_key():
    with open(API_KEY_FILE, 'r') as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise Exception("No API keys to rotate in api_key.txt")

    rotated_keys = keys[1:] + [keys[0]]
    with open(API_KEY_FILE, 'w') as f:
        for key in rotated_keys:
            f.write(key + '\n')
    logger.warning("🔁 Rotated API key. Retrying with a new key...")
    load_and_set_api_key()

# Initialize the API key and Gemini model
load_and_set_api_key()
llm = genai.GenerativeModel('gemini-2.0-flash')

# Template for image analysis
generic_template_freshness_detection = ''' 
You are a multimodal assistant contributing to a research initiative aimed at developing a large-scale, diverse, image-grounded question–answer dataset to train and evaluate Visual Question Answering (VQA) systems.

Your task is to analyze a given image and generate **12–15 unique and high-quality question–answer (QA) pairs**. Even if the image appears visually simple, sparse, or contains very few elements, you must still generate a minimum of 10 QA pairs by rephrasing, abstracting, or focusing on fine-grained visual cues.

You must pay **special attention to all visual indicators, including measurement labels, numerical values, scales, markers, lines, dimensions**, and any **clearly visible geometric or labeled cues**. These must always be incorporated into at least one question–answer pair.

---

**Question Requirements**

Each generated question must:

1.⁠ ⁠Be **entirely answerable from the image alone** — no commonsense reasoning, background knowledge, or external context.

2.⁠ ⁠Have a **single-word answer** — strictly a **noun, color, number, adjective, or verb**.

3.⁠ ⁠Avoid referencing any visible **text content or text labels** (no OCR-based questions like "What does the text say?"). However, **visible numeric or symbolic indicators of dimensions, sizes, or markers are permitted** and should be used.

4.⁠ ⁠Be **non-redundant** — do not ask similar questions about the same object, area, or visual property.

5.⁠ ⁠Be **clearly worded** and lead to a **visually unambiguous answer**.

6.⁠ ⁠Use a **diverse range of phrasings, question types, and difficulty levels** — from simple identification to more complex spatial and visual reasoning.

7. Explicitly include ** questions about any measurement, number, or dimension that is visually displayed** in the image (e.g., height, width, length, distances, angles, quantities, or values shown graphically or diagrammatically).

---

**Question Diversity Requirements**

Ensure coverage across a range of visual comprehension tasks:

- **Object Recognition** (e.g., “What is lying on the ground?” → “Ball”)
- **Color Identification** (e.g., “What color is the dress?” → “Blue”)
- **Counting** (e.g., “How many trees are there?” → “Two”)
- **Spatial Relationships** (e.g., “What is behind the chair?” → “Wall”)
- **Scene Understanding** (e.g., “What kind of place is this?” → “Market”)
- **Action Recognition** (e.g., “What is the boy doing?” → “Jumping”)
- **Facial Emotion** (e.g., “What is the emotion shown?” → “Surprise”)
- **Material Recognition** (e.g., “What is the table made of?” → “Glass”)
- **Environmental Cues** (e.g., “What is the weather like?” → “Rainy”, or “What time of day is shown?” → “Dusk”)
- **Measurement Identification** (e.g., “What is the width of the black object?” → “5cm” or “What is the height marked?” → “2in”)

---

**Output Format (.txt)**

Each QA pair must be recorded in the following format:

Qusetion: What is the animal doing?  
Answer: Sleeping

Qusetion: What color is the jacket?  
Answer: Yellow

Qusetion: How many chairs are visible?  
Answer: Four

Qusetion: What is next to the suitcase?  
Answer: Shoes

Qusetion: What emotion is the child showing?  
Answer: Joy

Qusetion: What is the height of the building  
Answer: 40 meters

Qusetion: What type of surface is the floor?  
Answer: Wood

Qusetion: What is the person holding?  
Answer: Camera

Qusetion: What time of day is it?  
Answer: Morning

---

**Important Notes**:

- Follow the exact format shown above.
- Maintain **maximum variation** in question type, phrasing, and visual reasoning.
- **Do not skip or reduce the question count** even with minimal visual information — use abstraction and rephrasing to extract at least 10 valid QA pairs.

'''

# Structured prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", generic_template_freshness_detection),
    ("user", "{text}")
])

# Output parser
parser = StrOutputParser()

def load_image_metadata(csv_path):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return [row for row in reader]
    except Exception as e:
        logger.error(f"Error reading metadata file: {e}")
        return []

def clean_text(text):
    return text.strip().strip('"').strip('\\')

def save_analysis_result(image_id, analysis_data, output_dir="analyzed_images"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        raw_data = json.dumps(analysis_data)
        blocks = raw_data.split('\\n\\n')

        qa_list = []
        for block in blocks:
            lines = block.strip().split('\\n')
            if len(lines) == 2:
                try:
                    question_line = lines[0].strip()
                    answer_line = lines[1].strip()
                    _, _, question_value = question_line.partition(':')
                    _, _, answer_value = answer_line.partition(':')
                    qa_list.append({
                        "question": clean_text(question_value),
                        "answer": clean_text(answer_value)
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Skipping block due to error: {block} — {e}")

        parsed_file = os.path.join(output_dir, f"{image_id}.json")
        with open(parsed_file, 'w') as outfile:
            json.dump(qa_list, outfile, indent=2)

        logger.info(f"✅ Parsed {len(qa_list)} Q&A pairs to: {parsed_file}")
    
    except Exception as e:
        logger.error(f"Error parsing and saving result for {image_id}: {e}")

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

def update_state(image_id, state_file="process_state.json"):
    try:
        state = {"last_processed_image": image_id}
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=4)
        logger.info(f"Updated process state with image_id {image_id}")
    except Exception as e:
        logger.error(f"Error updating state file: {e}")

def load_local_image(image_path):
    try:
        with Image.open(image_path) as img:
            img_byte_array = img.tobytes()
        return img_byte_array
    except Exception as e:
        logger.error(f"Error loading image from {image_path}: {e}")
        return None

def analyze_image(image_path: str):
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        mime_type = 'image/png' if image_path.lower().endswith('.png') else 'image/jpeg'
        if mime_type not in ['image/png', 'image/jpeg']:
            raise ValueError("Unsupported image format. Only PNG and JPEG are supported.")

        content_parts = [
            Part(text=generic_template_freshness_detection),
            Part(inline_data=Blob(mime_type=mime_type, data=image_data))
        ]

        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = llm.generate_content(Content(parts=content_parts), stream=True)
                response.resolve()
                parsed_response = parser.invoke(response.text)
                print(parsed_response)
                return parsed_response

            except Exception as e:
                error_msg = str(e).lower()
                known_errors = ["quota", "connection reset", "serviceunavailable", "deadline exceeded", "unavailable", "504", "503"]
                
                if any(known in error_msg for known in known_errors):
                    logger.warning(f"Retryable error: Retrying in 30 seconds... (Attempt {retry_count + 1}/{max_retries})")
                    time.sleep(30)
                    retry_count += 1
                else:
                    raise

        # After max retries, rotate API key and restart
        rotate_api_key()
        logger.info("🔁 Max retries hit. Rotating API key and restarting process...")
        subprocess.run(["python", "sample.py"])
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ An error occurred during image analysis: {e}")
        traceback.print_exc()
        # Always restart the process on unexpected errors too
        subprocess.run(["python", "sample.py"])
        sys.exit(0)


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
            analysis_result = analyze_image(image_path)
            save_analysis_result(image_id, analysis_result, output_dir)
            update_state(image_id)

if __name__ == "__main__":
    process_images()
