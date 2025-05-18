import os
import json

# Define the path to the directory containing JSON files
directory_path = './abo-listings_filtered/listings/metadata'

def filter_language_entries(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = filter_language_entries(value)
        return obj
    elif isinstance(obj, list):
        # If list items are dicts with 'language_tag', filter them
        if all(isinstance(item, dict) and 'language_tag' in item for item in obj):
            return [item for item in obj if item.get('language_tag') == 'es_US']
        else:
            return [filter_language_entries(item) for item in obj]
    else:
        return obj

# Iterate over each JSON file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        file_path = os.path.join(directory_path, filename)
        
        # Open and read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Iterate through each JSON object in the list
        for i, obj in enumerate(data):
            # Create the unified all_image_id field
            all_image_id = []
            
            if 'main_image_id' in obj:
                all_image_id.append(obj['main_image_id'])
            if 'other_image_id' in obj:
                all_image_id.extend(obj['other_image_id'])
            
            obj['all_image_id'] = all_image_id
            
            if 'main_image_id' in obj:
                del obj['main_image_id']
            if 'other_image_id' in obj:
                del obj['other_image_id']

            # Filter language-tagged list fields
            data[i] = filter_language_entries(obj)

        # Write the updated data back to the file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f'Processed {filename}')
