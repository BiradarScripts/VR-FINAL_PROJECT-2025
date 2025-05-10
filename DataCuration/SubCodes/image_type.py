import os
import json

# Input and output paths
input_folder = 'abo-listings_filtered/listings/metadata'
output_file = 'image_id_to_product_type.json'

# Dictionary to hold image_id: product_type pairs
image_id_map = {}

# Iterate through all .json files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.json'):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                product_type = item.get('product_type')
                image_ids = item.get('all_image_id', [])
                for image_id in image_ids:
                    image_id_map[image_id] = product_type

# Write the final mapping to a JSON file
with open(output_file, 'w', encoding='utf-8') as out_file:
    json.dump(image_id_map, out_file, indent=2)

print(f"Saved mapping of {len(image_id_map)} image IDs to {output_file}")
