# Bias Preparation Folder

## Overview
This subfolder, `bias_preparation`, is part of the AI project aimed at handling bias in datasets used for training machine learning models. It includes tools for preparing and verifying the data to ensure consistency across various biased subsets related to age and gender.

## Contents

- **biases/**: This directory contains duplicated images from the main dataset, structured into subdirectories based on two attributes: age and gender. The age-related subfolders are divided into `young`, `middle-aged`, and `old`. The gender-related folders are divided into `male` and `female`.

- **generate_bias_map_json.py**: A Python script that generates a `bias_map.json`. This JSON file maps each image file name to its corresponding age and gender attributes derived from the folder structure within `biases/`.

- **sync_datasets_img_names.py**: This script ensures the consistency of image filenames across the `biases/` folders and the main dataset folder (`./datasets`). It corrects any discrepancies in image names to align with those used in the root dataset folder, facilitating accurate bias mapping and analysis.

## Usage

### Generating Bias Map JSON
To create the bias map JSON file, which is crucial for subsequent bias analysis tasks, run:

```bash
python generate_bias_map_json.py
```

This script will traverse the `biases/` directory and generate a JSON file (`bias_map.json`) that lists each image along with its associated age and gender based on the folder structure.

### Synchronizing Dataset Image Names
If discrepancies are found between the names of images in the `data_preparation/biases/` folders and the main dataset `./dataset`, run:

```bash
python sync_datasets_img_names.py
```

This will adjust the names in the `biases/` folder to match the main dataset, ensuring that the mapping and analysis can be performed accurately.

## Notes
- It is crucial to run `sync_datasets_img_names.py` before `generate_bias_map_json.py` if there are any updates or changes to the main dataset folder.
- Make sure that the `biases/` folder structure is correctly maintained and that images are appropriately duplicated into respective age and gender folders for accurate bias analysis.

---