import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import math

# Define the input and output directories
input_dir = './abo-listings_filtered/listings/metadata'
output_dir = './output2'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory '{output_dir}' created (if it didn't exist).")

# Flag to track if any files were processed
processed_files = False

# A list to store the product types for analysis
product_types = []

# Iterate over each .json file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(input_dir, filename)
        
        # Read the content of the .json file
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print(f"Error reading {filename}, skipping.")
                continue
        
        # Check if data is a list or dictionary and process accordingly
        if isinstance(data, list):
            print(f"Contents of {filename}: This is a list, processing each item.")
            for idx, item in enumerate(data):
                if 'product_type' in item and item['product_type']:
                    product_type = item['product_type'][0]['value']  # Extract the product type value
                    
                    # If the product_type is empty, assign a default value
                    if not product_type:
                        product_type = "Unknown"
                    
                    # Add product type to the analysis list
                    product_types.append(product_type)
                    
                    # Create the directory for the product type if it doesn't exist
                    product_type_dir = os.path.join(output_dir, product_type)
                    os.makedirs(product_type_dir, exist_ok=True)
                    
                    # Define the output file path
                    new_filename = f"{os.path.splitext(filename)[0]}_{idx}.json"
                    output_file_path = os.path.join(product_type_dir, new_filename)
                    
                    # Save the item without removing 'product_type'
                    with open(output_file_path, 'w') as output_file:
                        json.dump(item, output_file, indent=4)
                    
                    print(f"Processed and moved {filename} item {idx} to {product_type_dir}")
                    processed_files = True
                else:
                    print(f"Skipping item {idx} in {filename}: 'product_type' field is missing or empty.")
        
        elif isinstance(data, dict):
            print(f"Contents of {filename}: This is a dictionary.")
            if 'product_type' in data and data['product_type']:
                product_type = data['product_type'][0]['value']
                
                if not product_type:
                    product_type = "Unknown"
                
                product_types.append(product_type)
                
                product_type_dir = os.path.join(output_dir, product_type)
                os.makedirs(product_type_dir, exist_ok=True)
                
                # Save the whole dict without removing 'product_type'
                output_file_path = os.path.join(product_type_dir, filename)
                
                with open(output_file_path, 'w') as output_file:
                    json.dump(data, output_file, indent=4)
                
                print(f"Processed and moved {filename} to {product_type_dir}")
                processed_files = True
            else:
                print(f"Skipping {filename}: 'product_type' field is missing or empty.")
        else:
            print(f"Skipping {filename}: The structure of the data is neither a dictionary nor a list.")

# Print a final message based on whether any files were processed
if not processed_files:
    print("No files were processed. Please check the input directory for valid .json files with the 'product_type' field.")
else:
    print("Processing complete.")

# Data Analysis and Graph Generation
if product_types:
    product_type_series = pd.Series(product_types)
    product_type_counts = product_type_series.value_counts()

    chunk_size = 50
    num_chunks = math.ceil(len(product_type_counts) / chunk_size)

    for i in range(num_chunks):
        start_index = i * chunk_size
        end_index = min((i + 1) * chunk_size, len(product_type_counts))
        chunk_data = product_type_counts[start_index:end_index]

        plt.figure(figsize=(12, 8))
        chunk_data.sort_values().plot(kind='barh', color='skyblue', edgecolor='black')

        plt.title(f'Distribution of Product Types (Part {i + 1})', fontsize=16)
        plt.xlabel('Frequency', fontsize=14)
        plt.ylabel('Product Type', fontsize=14)
        plt.tight_layout()

        for index, value in enumerate(chunk_data):
            plt.text(value, index, str(value), va='center', ha='left', fontweight='bold')

        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        graph_file_path = os.path.join(output_dir, f'product_type_distribution_part_{i + 1}.png')
        plt.savefig(graph_file_path)
        plt.close()
        
        print(f"Graph for part {i + 1} of product type distribution saved as '{graph_file_path}'.")

else:
    print("No product types found for analysis.")
