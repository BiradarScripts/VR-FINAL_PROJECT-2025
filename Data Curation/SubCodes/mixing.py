import os
import json

# Define the path to the directory containing JSON files
directory_path = './abo-listings_filtered/listings/metadata'

# Iterate over each JSON file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.json'):
        file_path = os.path.join(directory_path, filename)
        
        # Open and read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Iterate through each JSON object in the list
        for obj in data:
            # Create the unified all_image_id field
            all_image_id = []
            
            # Add both main_image_id and other_image_id to the list if they exist
            if 'main_image_id' in obj:
                all_image_id.append(obj['main_image_id'])
            if 'other_image_id' in obj:
                all_image_id.extend(obj['other_image_id'])  # If it's a list of IDs, merge them
            
            # Assign the unified list to all_image_id field
            obj['all_image_id'] = all_image_id
            
            # Remove the main_image_id and other_image_id fields
            if 'main_image_id' in obj:
                del obj['main_image_id']
            if 'other_image_id' in obj:
                del obj['other_image_id']
        
        # Write the updated data back to the file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        print(f'Processed {filename}')
