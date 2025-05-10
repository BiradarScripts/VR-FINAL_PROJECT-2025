import os
import json
import csv
import logging
import traceback
from dotenv import load_dotenv, set_key
from google.api_core.exceptions import ResourceExhausted
from google.ai.generativelanguage import Content, Part, Blob
import google.generativeai as genai
import subprocess
import sys
import time
from langchain_core.output_parsers import StrOutputParser
from PIL import Image

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_KEY_FILE = "api_key.txt"
ENV_FILE = ".env"
CHECKPOINT_FILE = "checkpoint.json"

# Load and set API key
def load_and_set_api_key():
    with open(API_KEY_FILE, 'r') as f:
        keys = [line.strip() for line in f if line.strip()]
    if not keys:
        raise Exception("No API keys found in api_key.txt")
    current_key = keys[0]
    set_key(ENV_FILE, "GOOGLE_API_KEY", current_key)
    genai.configure(api_key=current_key)
    return keys

# Rotate API key
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

# Initialize API
load_and_set_api_key()
llm = genai.GenerativeModel('gemini-2.0-flash')

# Template
generic_template_freshness_detection = '''
you are a multimodal assistant contributing to a research initiative aimed at building a large-scale, diverse, image-grounded Visual Question Answering (VQA) dataset.

Your task is:

Objective: Generate atleast 15 high-quality question-answer (QA) pairs per sample, derived from both visual content (image) and metadata (JSON object) associated with the image. The questions should explore various aspects of the product and ensure diversity in the types of questions asked. This will help create a robust dataset that ensures high model performance.

Input Guidelines:
Image: The visual content associated with the product.
Metadata: A structured JSON object that contains various product details like brand, color, bullet points, material, etc.,remember zero or more fileds
might not be present,look very careful in the bullet_point,product_description,product_type section of the metadata,to obtain crucial information about the product.very important
go through all the fields in detailed and then form question answer pair.
Example Metadata Format:

    - `brand`
        - Content: Brand name
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `bullet_point`
        - Content: Important features of the products
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `color`
        - Content: Color of the product as text
        - Format: `[{"language_tag": <str>, "standardized_values": [<str>],
          "value": <str>}, ...]`
    - `color_code`
        - Content: Color of the product as HTML color code
        - Format: `[<str>, ...]`
         - `fabric_type`
        - Content: Description of product fabric
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `finish_type`
        - Content: Description of product finish
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `item_shape`
        - Content: Description of the product shape
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `material`
        - Content: Description of the product material
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `pattern`
        - Content: Product pattern
        - Format: `[{ "language_tag": <str>, "value": <int> }, ...]`
    - `product_description`
        - Content: Product description as HTML 
        - Format: `[{ "language_tag": <str>, "value": <int> }, ...]`
    - `product_type`
        - Content: Product type (category)
        - Format: `<str>`
         - `style`
        - Content: Style of the product
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`

also be very careful with the language_tag.


**Question Requirements**
Each generated question must:
1.⁠ ⁠Be **entirely answerable from the image  only** — no commonsense reasoning, background knowledge, or external context.
2.⁠ ⁠Have a **single-word answer** — strictly a **noun, color, number, adjective, or verb**.
3.⁠ ⁠Be **non-redundant** — do not ask similar questions about the same object, area, or visual property.
4.⁠ ⁠Be **clearly worded** and lead to a **visually unambiguous answer**.
5.⁠ ⁠Use a **diverse range of phrasings, question types, and difficulty levels** — from simple identification to more complex spatial and visual reasoning.
6. Explicitly include numerical ** questions about any measurement, number, or dimension that if visually  present or visible provided to you in the image ** in the image (e.g., height, width, length, distances, angles, quantities, or values shown graphically or diagrammatically) .
7.DOnt consider questions whose answers are not visible in the questino for example , if a box is given dont ask question how many packets are there in the box,as the packets are not visible in a 2d image
---

Guidelines for Question Generation:
Visual Content-Based Questions: These questions should focus on elements visible in the image. They must be related to the product’s appearance, object recognition, and scene understanding.
Questions should also be based on the relation between objects
Metadata-Based Questions: These questions should focus on the information available in the metadata and should not overlap with visual content unless explicitly shown. They can ask about the product’s brand, dimensions, weight, color, etc.
Types of Questions:
Object Recognition: Identifying visible objects or items.
Color Identification: Asking about the color of visible elements or based on metadata.
Counting: Counting items, features, or objects in the image.
Spatial Relationships: Questions regarding the positioning or relation of objects in the image.
Material Recognition: Questions about the material based on metadata.
Action Recognition: Any observable activity or usage depicted in the image.
Bullet Point Understanding: Questions based on the product’s important features as listed in the metadata.
Question Diversity: Ensure at least 6 questions focus on visual content and at least 6 focus on metadata. The questions should cover a wide variety of comprehension skills.
Output Format:
For each sample, provide atleast 15 unique question-answer pairs:

At least 5-6 questions based on visual content.
At least 5-6 questions based on metadata content.
NO boolean based questions,no yes/no based answer.

---
**Output Format (.txt)**

Qusetion: What is the color of the product?
Answer: Black

Qusetion: What is the product made of?
Answer: Mesh

Qusetion:what is in front of the door and on the right of the table in the image5 ?
Answer:chair

Question:what color is the wall behind the projector screen in the image ?
Answer:black

Question:what is the brown object above the file cabinet in the image?
Answer:bookShelf

Qusetion: How many items are visible in the image?
Answer: 10

Qusetion: what color is the ladder between the projector screen and armchairs in the image?
Answer: red

Qusetion: What is the brand of the product?
Answer: Nike

Qusetion: What style is the product?
Answer: Sporty

Qusetion: what is on the left side of the white oven on the floor and on right side of the blue armchair in the image ?
Answer: garbage_bin

Qusetion: What type of product is shown in the image?
Answer: Shoes

Qusetion: What is the material of the product?
Answer: Rubber

Qusetion: What is the height of the product given in the image?
Answer: 30 cm

Qusetion: Describe the pattern of the product.
Answer: None

Qusetion: What is the shape of the product?
Answer: Rectangular

Qusetion: what is on the right side of the notebook on the desk in the image?
Answer: plastic_cup_of_coffee
---
**Important Notes**:
-Be very careful in the bullet_point section of the metadata,the dimensions of the object might be given there aswell,and also many other very cruicial information can be extracted from there to form a question/answer pair
-only take into consideration the information from the fields given in the metadata, dont assume anything,else world would collapse
- Follow the exact format shown above.
- Maintain **maximum variation** in question type, phrasing, and visual reasoning.
- **Do not skip or reduce the question count** even with minimal visual information — use abstraction and rephrasing to extract at least 15 valid QA pairs.
- generate very high quality state of the art Qusetion answer pair,the world needs it,its th eneed of the hour
'''

# Output parser
parser = StrOutputParser()

def clean_text(text):
    return text.strip().strip('"').strip('\\')


def save_analysis_result(image_id, analysis_data, image_path, output_dir="analyzed_images"):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert analysis data to JSON string and split into blocks
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

        # Save analysis data along with image path
        parsed_file = os.path.join(output_dir, f"{image_id}.json")
        result_data = {
            "image_path": image_path,  # Add the image path to the result
            "qa_pairs": qa_list  # Include the Q&A pairs
        }
        
        # Write the result to a JSON file
        with open(parsed_file, 'w') as outfile:
            json.dump(result_data, outfile, indent=2)

        logger.info(f"✅ Parsed {len(qa_list)} Q&A pairs and saved to: {parsed_file}")
    
    except Exception as e:
        logger.error(f"Error parsing and saving result for {image_id}: {e}")


# Load image metadata
def load_image_metadata(csv_path):
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return {row["image_id"]: row["path"] for row in reader}
    except Exception as e:
        logger.error(f"Error reading metadata file: {e}")
        return {}

# Analyze single image with context
def analyze_image_with_context(image_path, json_context):
    try:
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        mime_type = 'image/png' if image_path.lower().endswith('.png') else 'image/jpeg'
        if mime_type not in ['image/png', 'image/jpeg']:
            raise ValueError("Unsupported image format. Only PNG and JPEG are supported.")

        content_parts = [
            Part(text=generic_template_freshness_detection),
            Part(text=json.dumps(json_context, indent=2)),
            Part(inline_data=Blob(mime_type=mime_type, data=image_data))
        ]

        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = llm.generate_content(Content(parts=content_parts), stream=True)
                response.resolve()
                parsed_response = parser.invoke(response.text)
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

        rotate_api_key()
        logger.info("🔁 Max retries hit. Rotating API key and restarting process...")
        subprocess.run(["python", "final.py"])
        sys.exit(0)

    except Exception as e:
        logger.error(f"❌ An error occurred during image analysis: {e}")
        traceback.print_exc()
        subprocess.run(["python", "final.py"])
        sys.exit(0)

# Load checkpoint
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
    return set()

# Save checkpoint
def save_checkpoint(processed_ids):
    try:
        with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(processed_ids), f, indent=2)
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")

# Main processing function
def process_json_objects(base_folder="part_1", csv_path="abo-images-small/images/metadata/images.csv", output_dir="analyzed_images2"):
    image_metadata = load_image_metadata(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    processed_ids = load_checkpoint()

    # Get the subfolders in the base folder in sequential order (lexicographically sorted)
    subfolders = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])

    for subfolder in subfolders:
        subfolder_path = os.path.join(base_folder, subfolder)
        for root, dirs, files in os.walk(subfolder_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            try:
                                obj = json.load(f)  # Single JSON object per file
                            except json.JSONDecodeError as e:
                                logger.error(f"⚠️ JSON decoding error in {file_path}: {e}")
                                continue  # skip this file

                        image_ids = obj.get("all_image_id", [])
                        context = {k: v for k, v in obj.items() if k != "all_image_id" and k!='item_weight' and k!='item_dimensions'}

                        for image_id in image_ids:
                            if image_id in processed_ids:
                                logger.info(f"✅ Skipping already processed image {image_id}")
                                continue

                            image_rel_path = image_metadata.get(image_id)
                            if not image_rel_path:
                                logger.warning(f"⚠️ Image ID {image_id} not found in metadata.")
                                continue

                            image_full_path = os.path.join("abo-images-small", "images", "small", image_rel_path)
                            if not os.path.exists(image_full_path):
                                logger.warning(f"⚠️ Image path {image_full_path} does not exist.")
                                continue

                            logger.info(f"Processing image {image_id} from {image_full_path} with context from {file_path}")

                            analysis_result = analyze_image_with_context(image_full_path, context)

                            save_analysis_result(image_id, analysis_result, image_full_path, output_dir)

                            processed_ids.add(image_id)
                            save_checkpoint(processed_ids)

                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
                        traceback.print_exc()


    


if __name__ == "__main__":
    process_json_objects()