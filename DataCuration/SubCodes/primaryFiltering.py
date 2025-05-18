import os
import json

# Define the folder path and the fields you want to keep
input_folder = './abo-listings/listings/metadata'
output_folder = './abo-listings_filtered/listings/metadata'  # create this if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

fields_to_keep = {
    "bullet_point",
    "color",
    "color_code",
    "fabric_type",
    "finish_type",
    "item_dimensions",
    "item_shape",
    "item_weight",
    "material",
    "pattern",
    "product_description",
    "product_type",
    "style",
}


# Iterate through all JSON files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith('.json'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        with open(input_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

            def filter_object(obj):
                new_obj = {key: obj[key] for key in fields_to_keep if key in obj}
                # Merge image IDs
                all_images = []
                if 'main_image_id' in obj:
                    all_images.append(obj['main_image_id'])
                if 'other_image_id' in obj:
                    if isinstance(obj['other_image_id'], list):
                        all_images.extend(obj['other_image_id'])
                    else:
                        all_images.append(obj['other_image_id'])
                if all_images:
                    new_obj['all_image_ids'] = all_images
                return new_obj

            if isinstance(data, list):
                filtered_data = [filter_object(obj) for obj in data]
            else:
                filtered_data = filter_object(data)

        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(filtered_data, outfile, indent=2)

print("Filtered JSON files saved in:", output_folder)
