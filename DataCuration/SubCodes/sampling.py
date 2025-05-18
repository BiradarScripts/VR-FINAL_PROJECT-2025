import os
import json
import math
import random


SOURCE_DIR = "./output2"
DEST_DIR = "./sampled_output"

ALPHA = 0.5
MIN_SAMPLES = 1
MIN_FIXED_TIER4 = 100
MAX_SAMPLES = 15000

def determine_tier(size):
    if size >= 100_000:
        return 1, 0.05
    elif size >= 10_000:
        return 2, 0.20
    elif size >= 1_000:
        return 3, 0.50
    else:
        return 4, 1.00

def compute_sample_count(size, tier, ratio):
    if tier == 4 and size < MIN_FIXED_TIER4:
        return MIN_FIXED_TIER4
    scaled = (math.log10(size) * ALPHA + ratio) * size
    return int(min(max(MIN_SAMPLES, scaled), MAX_SAMPLES))

def get_all_image_ids_from_file(file_path):
    image_ids = []
    try:
        with open(file_path, 'r') as f:
            lines = f.read().strip().splitlines()
            for line in lines:
                if line.strip():
                    obj = json.loads(line)
                    if 'all_image_id' in obj:
                        image_ids.extend(obj['all_image_id'])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return image_ids

def write_sampled_json(file_path, sampled_ids):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump({"sampled_image_ids": sampled_ids}, f, indent=2)
    except Exception as e:
        print(f"Error writing {file_path}: {e}")

# ------------------------- MAIN -------------------------------

def proportional_sampling():
    print("Starting proportional sampling...")

    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if not file.endswith(".json"):
                continue

            source_path = os.path.join(root, file)
            relative_path = os.path.relpath(source_path, SOURCE_DIR)
            dest_path = os.path.join(DEST_DIR, relative_path)

            image_ids = get_all_image_ids_from_file(source_path)
            size = len(image_ids)

            if size == 0:
                print(f"Skipping empty file: {source_path}")
                continue

            tier, ratio = determine_tier(size)
            sample_count = compute_sample_count(size, tier, ratio)

            if tier != 4 or size >= MIN_FIXED_TIER4:
                sampled = random.sample(image_ids, min(sample_count, size))
            else:
                times = math.ceil(MIN_FIXED_TIER4 / size)
                sampled = (image_ids * times)[:MIN_FIXED_TIER4]

            write_sampled_json(dest_path, sampled)

    print("✅ Sampling complete! Check folder: sampled_output")

if __name__ == "__main__":
    proportional_sampling()
