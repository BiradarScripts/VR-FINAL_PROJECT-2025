# import os
# import shutil

# batches_dir = './Batches'

# # Traverse all subdirectories
# for root, dirs, files in os.walk(batches_dir):
#     for file in files:
#         if file.endswith('.json'):
#             source_path = os.path.join(root, file)
#             dest_path = os.path.join(batches_dir, file)
            
#             # Rename if a file with the same name already exists in the destination
#             if os.path.exists(dest_path):
#                 base, ext = os.path.splitext(file)
#                 i = 1
#                 while os.path.exists(os.path.join(batches_dir, f"{base}_{i}{ext}")):
#                     i += 1
#                 dest_path = os.path.join(batches_dir, f"{base}_{i}{ext}")
            
#             shutil.move(source_path, dest_path)

# print("All .json files moved to the Batches folder.")

# import os

# batches_dir = './Batches'

# json_files = [f for f in os.listdir(batches_dir) if f.endswith('.json') and os.path.isfile(os.path.join(batches_dir, f))]

# print(f"Total .json files in '{batches_dir}': {len(json_files)}")

# import os
# import json
# import shutil
# from tqdm import tqdm

# # Paths
# batches_dir = "Batches"
# image_map_path = "image_id_to_product_type.json"
# excess_dir = "Excess_CELLULAR_PHONE_CASE"

# # Create the excess folder if it doesn't exist
# os.makedirs(excess_dir, exist_ok=True)

# # Load image_id to product_type mapping
# with open(image_map_path, 'r') as f:
#     image_id_to_product_type = json.load(f)

# # Extract image_ids of CELLULAR_PHONE_CASE
# cellular_case_ids = [
#     image_id for image_id, product_list in image_id_to_product_type.items()
#     if product_list[0]["value"] == "CELLULAR_PHONE_CASE"
# ]

# # Limit to 15,000 to keep
# cellular_case_ids_to_keep = set(cellular_case_ids[:14758])

# # Move excess files
# for image_id in tqdm(cellular_case_ids[14758:], desc="Moving excess files"):
#     file_name = f"{image_id}.json"
#     src = os.path.join(batches_dir, file_name)
#     dst = os.path.join(excess_dir, file_name)
#     if os.path.exists(src):
#         shutil.move(src, dst)

import os
import shutil
from tqdm import tqdm

# Paths
excess_dir = "Excess_CELLULAR_PHONE_CASE"
batches_dir = "Batches"

# Ensure destination folder exists
os.makedirs(batches_dir, exist_ok=True)

# Get list of all JSON files in Excess folder
json_files = [f for f in os.listdir(excess_dir) if f.endswith(".json")]

# Limit to 8000 files
files_to_move = json_files[:8000]

# Move files
for file_name in tqdm(files_to_move, desc="Moving files to Batches"):
    src = os.path.join(excess_dir, file_name)
    dst = os.path.join(batches_dir, file_name)
    if os.path.exists(src):
        shutil.move(src, dst)
