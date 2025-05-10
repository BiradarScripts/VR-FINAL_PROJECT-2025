import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt

# Load image_id to product_type mapping
with open('image_id_to_product_type.json', 'r') as f:
    image_id_map = json.load(f)

# Path to Batches folder
batches_folder = 'Batches'

# Initialize counter
category_counts = defaultdict(int)

# List all .json files in Batches
for filename in os.listdir(batches_folder):
    if filename.endswith('.json'):
        image_id = os.path.splitext(filename)[0]
        if image_id in image_id_map:
            product_info = image_id_map[image_id]
            if product_info and isinstance(product_info, list):
                product_type = product_info[0].get("value")
                if product_type:
                    category_counts[product_type] += 1

# Sort categories by count in descending order
sorted_counts = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

# Total images and number of product types
total_count = sum(category_counts.values())
num_product_types = len(category_counts)

# Write to .txt file
output_file = 'category_counts.txt'
with open(output_file, 'w') as f:
    f.write("Category-wise image count (sorted by count):\n")
    for category, count in sorted_counts:
        f.write(f"{category}: {count}\n")
    f.write(f"\nTotal Images: {total_count}\n")
    f.write(f"Number of Unique Product Types: {num_product_types}\n")

# Plotting the graph
categories = [item[0] for item in sorted_counts]
counts = [item[1] for item in sorted_counts]

plt.figure(figsize=(12, 8))  # Increase the size for better visibility
bars = plt.barh(categories, counts, color='skyblue')
plt.xlabel('Image Count')
plt.ylabel('Product Type')
plt.title(f'Image Count per Product Type (Total: {total_count}, Types: {num_product_types})')
plt.gca().invert_yaxis()  # Highest count at the top
plt.tight_layout()

# Add value labels to bars
for bar in bars:
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{int(bar.get_width())}', va='center')

# Save the plot
plt.savefig('category_counts.png', dpi=300)  # High resolution
plt.close()

# Console output
print("Category-wise image count (sorted by count):")
for category, count in sorted_counts:
    print(f"{category}: {count}")
print(f"\nTotal Images: {total_count}")
print(f"Number of Unique Product Types: {num_product_types}")
