import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.ai.generativelanguage import Content, Part, Blob
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Retrieve API key from environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise Exception("API key not found. Please set the GOOGLE_API_KEY environment variable.")
else:
    genai.configure(api_key=API_KEY)  # Configure the API key

# Initialize the Google Gemini model
llm = genai.GenerativeModel('gemini-2.0-flash')

# The template for generating the response
generic_template_freshness_detection = '''
you are a multimodal assistant contributing to a research initiative aimed at building a large-scale, diverse, image-grounded Visual Question Answering (VQA) dataset.

Your task is:

Objective: Generate atleast 13 high-quality question-answer (QA) pairs per sample, derived from both visual content (image) and metadata (JSON object) associated with the image. The questions should explore various aspects of the product and ensure diversity in the types of questions asked. This will help create a robust dataset that ensures high model performance.

Input Guidelines:
Image: The visual content associated with the product.
Metadata: A structured JSON object that contains various product details like brand, color, dimensions, weight, bullet points, material, etc.,remember zero or more fileds
might not be present
Example Metadata Format:

{
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
    - `item_dimensions`
        - Content: Dimensions of the product (height, width, length)
        - Format: `{"height": {"normalized_value": {"unit": <str>, "value":
          <float>}, "unit": <str>, "value": <float>}, "length":
          {"normalized_value": {"unit": <str>, "value": <float>}, "unit": <str>,
          "value": <float>}, "width": {"normalized_value": {"unit": <str>,
          "value": <float>}, "unit": <str>, "value": <float>}}}`
           - `item_name`
        - Content: The product name
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `item_shape`
        - Content: Description of the product shape
        - Format: `[{ "language_tag": <str>, "value": <str> }, ...]`
    - `item_weight`
        - Content: The product weight
        - Format: `[{"normalized_value": {"unit": <str>, "value": <float>},
          "unit": <str>, "value": <float>}, ...]`
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


**Question Requirements**
Each generated question must:
1.⁠ ⁠Be **entirely answerable from the image alone** — no commonsense reasoning, background knowledge, or external context.
2.⁠ ⁠Have a **single-word answer** — strictly a **noun, color, number, adjective, or verb**.
3.⁠ ⁠Be **non-redundant** — do not ask similar questions about the same object, area, or visual property.
4.⁠ ⁠Be **clearly worded** and lead to a **visually unambiguous answer**.
5.⁠ ⁠Use a **diverse range of phrasings, question types, and difficulty levels** — from simple identification to more complex spatial and visual reasoning.
6. Explicitly include ** questions about any measurement, number, or dimension that is visually displayed** in the image (e.g., height, width, length, distances, angles, quantities, or values shown graphically or diagrammatically).
---

Guidelines for Question Generation:
Visual Content-Based Questions: These questions should focus on elements visible in the image. They must be related to the product’s appearance, object recognition, and scene understanding.
Metadata-Based Questions: These questions should focus on the information available in the metadata and should not overlap with visual content unless explicitly shown. They can ask about the product’s brand, dimensions, weight, color, etc.
Types of Questions:
Object Recognition: Identifying visible objects or items.
Color Identification: Asking about the color of visible elements or based on metadata.
Counting: Counting items, features, or objects in the image.
Spatial Relationships: Questions regarding the positioning or relation of objects in the image.
Material Recognition: Questions about the material based on metadata.
Action Recognition: Any observable activity or usage depicted in the image.
Dimensions/Measurement: Questions related to the product’s size, weight, and other metrics.
Bullet Point Understanding: Questions based on the product’s important features as listed in the metadata.
Question Diversity: Ensure at least 6 questions focus on visual content and at least 6 focus on metadata. The questions should cover a wide variety of comprehension skills.
Output Format:
For each sample, provide atleast 13 unique question-answer pairs:

At least 6 questions based on visual content.
At least 6 questions based on metadata content.
Qusetion: What is the color of the product?  
Answer: Black

Qusetion: What is the product made of?  
Answer: Mesh

Qusetion: How many items are visible in the image?  
Answer: One

Qusetion: What is the brand of the product?  
Answer: Nike

Qusetion: What is the weight of the product?  
Answer: 1.2 kg

Qusetion: What style is the product?  
Answer: Sporty

Qusetion: What is the length of the product?  
Answer: 50 cm

Qusetion: What type of product is shown in the image?  
Answer: Shoes

Qusetion: Is the product's material rubber?  
Answer: Yes

Qusetion: Is the product lightweight?  
Answer: Yes

Qusetion: Is there any visible logo on the product?  
Answer: Yes

Qusetion: What is the height of the product?  
Answer: 30 cm

Qusetion: Does the product have a pattern?  
Answer: No

Qusetion: What is the shape of the product?  
Answer: Rectangular

Qusetion: Does the product have a breathable fabric?  
Answer: Yes

**Important Notes**:
-only take into consideration the information from the fields given in the metadata, dont assume anything,else world would collapse
- Follow the exact format shown above.
- Maintain **maximum variation** in question type, phrasing, and visual reasoning.
- **Do not skip or reduce the question count** even with minimal visual information — use abstraction and rephrasing to extract at least 13 valid QA pairs.


'''
# Create a structured prompt using ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template_freshness_detection),
        ("user", "{text}")
    ]
)

# Initialize the output parser
parser = StrOutputParser()

# Prepare the content parts for image and metadata analysis
def analyze_image_with_metadata(image_path: str, metadata: str):
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

        # Prepare content parts with system instructions, image, and metadata separately
        content_parts = [
            Part(text=generic_template_freshness_detection),  # System message
            Part(inline_data=Blob(mime_type=mime_type, data=image_data)),  # The image
            Part(text=f"Metadata information:\n{metadata}")  # Metadata separately
        ]

        # Generate the content (stream=True for real-time generation)
        response = llm.generate_content(Content(parts=content_parts), stream=True)
        response.resolve()

        # Parse the AI's analysis result
        parsed_response = parser.invoke(response.text)

        # Return the parsed result
        return parsed_response

    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e)

# Example usage with the provided image and metadata
image_path = './image.png'  # Replace with actual path to image
metadata_text = '''
"product_type": [
        {
            "value": "PRESSURE_COOKER"
        }
    ],
    "style": [
        {
            "language_tag": "en_AU",
            "value": "4 L"
        }
    ],
    "bullet_point": [
        {
            "language_tag": "en_AU",
            "value": "Stainless-steel pressure cooker for cooking delicious, nutritious meals in less time"
        },
        {
            "language_tag": "en_AU",
            "value": "2 cooking pressure levels: 50 kPa (gentle cooking) and 90 kPa (fast cooking)"
        },
        {
            "language_tag": "en_AU",
            "value": "Water-level indicator for proper filling; securely connected Bakelite handles; quiet operation"
        },
        {
            "language_tag": "en_AU",
            "value": "Dishwasher-safe (except for lid); suitable for all heat sources, including induction, flame, and ceramic stove tops"
        },
        {
            "language_tag": "en_AU",
            "value": "Convenient 4-liter size; measures 22 cm in diameter; pressure-indicator valve for safe opening"
        }
    ],
    "item_dimensions": {
        "height": {
            "normalized_value": {
                "unit": "inches",
                "value": 6.5
            },
            "unit": "inches",
            "value": 6.5
        },
        "length": {
            "normalized_value": {
                "unit": "inches",
                "value": 9.92
            },
            "unit": "inches",
            "value": 9.92
        },
        "width": {
            "normalized_value": {
                "unit": "inches",
                "value": 17.32
            },
            "unit": "inches",
            "value": 17.32
        }
    },
    "brand": [
        {
            "language_tag": "en_AU",
            "value": "AmazonBasics"
        }
    ],
    "item_name": [
        {
            "language_tag": "en_AU",
            "value": "Amazonbasics Stainless Steel Pressure Cooker, 4 L"
        }
    ],
    "item_weight": [
        {
            "normalized_value": {
                "unit": "pounds",
                "value": 5.76
            },
            "unit": "pounds",
            "value": 5.76
        }
    ]
'''

result = analyze_image_with_metadata(image_path, metadata_text)
print(result)
