# import os
# import json
# import shutil
# from collections import defaultdict

# # Input paths
# batches_folder = 'Batches'
# mapping_file = 'image_id_to_product_type.json'

# # Load mapping from image_id to product_type
# with open(mapping_file, 'r') as f:
#     mapping_data = json.load(f)

# # Convert mapping: image_id -> product_type
# id_to_type = {
#     image_id: entry[0]['value']
#     for image_id, entry in mapping_data.items()
# }

# # Track file count per product_type
# product_type_counts = defaultdict(int)

# # Process each .json in Batches
# for filename in os.listdir(batches_folder):
#     if filename.endswith('.json'):
#         image_id = os.path.splitext(filename)[0]
#         if image_id in id_to_type:
#             product_type = id_to_type[image_id]
#             dest_folder = os.path.join(batches_folder, product_type)
#             os.makedirs(dest_folder, exist_ok=True)

#             src_path = os.path.join(batches_folder, filename)
#             dest_path = os.path.join(dest_folder, filename)

#             shutil.move(src_path, dest_path)
#             product_type_counts[product_type] += 1
#         else:
#             print(f"⚠️ Warning: {image_id} not found in mapping file.")

# # Print results
# print("\n📦 Files moved per product_type folder:")
# for product_type, count in sorted(product_type_counts.items()):
#     print(f"{product_type}: {count} files")


# import os

# def count_json_files_and_folders(root_folder):
#     json_count = 0
#     folder_count = 0

#     for dirpath, dirnames, filenames in os.walk(root_folder):
#         folder_count += 1  # Count current folder
#         json_count += sum(1 for file in filenames if file.endswith('.json'))

#     return json_count, folder_count

# # Replace 'Batches' with your actual folder path if needed
# folder_path = 'Batches'
# json_count, folder_count = count_json_files_and_folders(folder_path)

# print(f"Total JSON files in '{folder_path}': {json_count}")
# print(f"Total folders (including root) in '{folder_path}': {folder_count}")

# import os

# # Path to the parent directory
# parent_directory = 'Batches'

# # Initialize counters for each range of 10 .json files
# counts = {i: 0 for i in range(0, 101, 10)}

# # Traverse through the subdirectories
# for subfolder in os.listdir(parent_directory):
#     subfolder_path = os.path.join(parent_directory, subfolder)
    
#     # Only proceed if it's a directory
#     if os.path.isdir(subfolder_path):
#         # Get all .json files in the subfolder
#         json_files = [f for f in os.listdir(subfolder_path) if f.endswith('.json')]
#         num_files = len(json_files)
        
#         # Count the folders for specific ranges (0-9, 10-19, ..., 90-99)
#         if 0 <= num_files <= 100:
#             range_key = (num_files // 10) * 10
#             counts[range_key] += 1

# # Calculate the cumulative sum in increasing order
# cumulative_sum = 0
# for range_key in sorted(counts.keys()):
#     cumulative_sum += counts[range_key]
#     print(f"Folders with {range_key}-{range_key+9} .json files: {counts[range_key]}")
#     print(f"Cumulative sum: {cumulative_sum}")



# import os
# import shutil
# import random
# from pathlib import Path

# # Paths
# source_root = Path("Batches")
# test_root = Path("test_dataset")
# test_root.mkdir(exist_ok=True)

# # Walk through subfolders
# for class_dir in source_root.iterdir():
#     if class_dir.is_dir():
#         json_files = list(class_dir.glob("*.json"))

#         # Skip folders with fewer than 20 .json files
#         if len(json_files) < 20:
#             print(f"Skipping '{class_dir.name}' (only {len(json_files)} JSON files)")
#             continue

#         # Select 20% of the files randomly
#         test_count = max(1, int(0.2 * len(json_files)))
#         selected_files = random.sample(json_files, test_count)

#         for file_path in selected_files:
#             target_path = test_root / file_path.name

#             # Skip if file with same name already exists
#             if target_path.exists():
#                 print(f"Skipping '{file_path.name}' (already exists in test_dataset/)")
#                 continue

#             shutil.move(str(file_path), str(target_path))

#         print(f"Moved {test_count} files from '{class_dir.name}' to test_dataset/")

# import os

# folder_path = "./test_dataset"  

# json_count = sum(1 for file in os.listdir(folder_path) if file.endswith(".json"))

# print(f"Total .json files in '{folder_path}': {json_count}")

import os
import shutil
import random
from pathlib import Path

# Define paths
source_root = Path("Batches")
master_train_root = Path("master_train")
master_train_root.mkdir(exist_ok=True)

# Collect all .json files from each class/sub-folder
class_jsons = {}
for class_dir in source_root.iterdir():
    if class_dir.is_dir():
        json_files = list(class_dir.glob("*.json"))
        if json_files:
            class_jsons[class_dir.name] = json_files

# Shuffle each class's files to ensure randomness within each class
for class_name in class_jsons:
    random.shuffle(class_jsons[class_name])

# Define batch size (10,000 items)
batch_size = 10000
batch_number = 1
current_batch = []

# Create batch folders and distribute files
while any(class_jsons.values()):  # While there are still files to move
    # Ensure each class contributes to the current batch
    for class_name, json_files in list(class_jsons.items()):
        if json_files:
            current_batch.append(json_files.pop(0))  # Pop the first file of each class

        # If all files from a class are moved, remove that class from the list
        if not json_files:
            del class_jsons[class_name]

    # Once batch reaches batch size, move files and create new batch folder
    if len(current_batch) >= batch_size:
        batch_folder = master_train_root / f"batch_{batch_number}"
        batch_folder.mkdir(exist_ok=True)

        # Move files in the current batch to the batch folder
        for json_file in current_batch:
            shutil.move(str(json_file), str(batch_folder / json_file.name))

        print(f"Batch {batch_number} moved with {len(current_batch)} files.")

        # Reset for the next batch
        batch_number += 1
        current_batch = []

# If there are any remaining files (less than 10,000), move them into the last batch
if current_batch:
    batch_folder = master_train_root / f"batch_{batch_number}"
    batch_folder.mkdir(exist_ok=True)

    for json_file in current_batch:
        shutil.move(str(json_file), str(batch_folder / json_file.name))

    print(f"Batch {batch_number} moved with {len(current_batch)} files.")
