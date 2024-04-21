import os
import json

def create_bias_map(root_dir):
    bias_map = {}
    extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}  # Support both cases and additional JPEG

    # Function to add files to the bias map
    def add_to_map(file, category, category_type):
        if any(file.endswith(ext) for ext in extensions):
            if file not in bias_map:
                bias_map[file] = {}
            bias_map[file][category_type] = category

    # Walk through all subdirectories under each category type
    for category_type in ['age', 'gender']:
        category_dir = os.path.join(root_dir, category_type)
        for root, dirs, files in os.walk(category_dir):
            category = os.path.basename(root)  # The category is the current directory's name
            for file in files:
                try:
                    add_to_map(file, category, category_type)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    # Write the dictionary to a JSON file
    with open('bias_map.json', 'w') as json_file:
        json.dump(bias_map, json_file, indent=4)

    return bias_map

# Usage
root_directory = 'biases'  # Adjust to the correct path
create_bias_map(root_directory)