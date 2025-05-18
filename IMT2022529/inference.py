import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import os

from transformers import BlipProcessor, BlipForQuestionAnswering
from peft import PeftModel, PeftConfig

def main():
    parser = argparse.ArgumentParser(description='Run inference on a BLIP VQA model using images and questions from a CSV.')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to the directory containing images.')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file with image names and questions.')
    args = parser.parse_args()

    MODEL_PATH = "aryamanpathak/blip-vqa-abo"
    MAX_ANSWER_LENGTH = 20

    # Load CSV
    try:
        df = pd.read_csv(args.csv_path)
        if 'image_name' not in df.columns or 'question' not in df.columns:
            raise ValueError("CSV must contain 'image_name' and 'question' columns.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # ⚡ Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load model + PEFT adapter
        peft_config = PeftConfig.from_pretrained(MODEL_PATH)
        base_model = BlipForQuestionAnswering.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)

        # ⚡ Model to GPU
        model.to(device)

        # ⚡ Optional speed-up with PyTorch 2.0+ (graph compilation)
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile()")
        except Exception as e:
            print(f"torch.compile() not applied: {e}")

        processor = BlipProcessor.from_pretrained(peft_config.base_model_name_or_path)
    except Exception as e:
        print(f"Error loading model or processor from {MODEL_PATH}: {e}")
        return

    model.eval()  # Set model to eval mode

    generated_answers = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating answers"):
        image_path = os.path.join(args.image_dir, row['image_name'])
        question = str(row['question'])
        answer = "error"

        try:
            image = Image.open(image_path).convert("RGB")
            # Preprocess
            encoding = processor(image, question, return_tensors="pt")
            encoding = {k: v.to(device) for k, v in encoding.items()}  # ⚡ Move inputs to GPU

            # ⚡ Use inference_mode() for faster forward pass (no autograd)
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids=encoding["input_ids"],
                    pixel_values=encoding["pixel_values"],
                    attention_mask=encoding.get("attention_mask", None),
                    max_length=MAX_ANSWER_LENGTH
                )

            # Decode answer
            answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        except FileNotFoundError:
            answer = f"Error: Image file not found at {image_path}"
        except Exception as e:
            answer = f"Error processing sample {idx}: {e}"

        answer = str(answer).strip().lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    output_csv_path = "results.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Generated answers saved to {output_csv_path}")


if __name__ == "__main__":
    main()