import os
from PIL import Image
import imagehash

def hash_images(directory, allowed_extensions=('.jpg', '.png', '.jpeg')):
    """Hash images in a directory and return a dictionary of hashes to filenames."""
    image_hashes = {}
    print(f"Hashing images in {directory}...")
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(allowed_extensions) and not file.startswith('.'):
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as img:
                        img_hash = str(imagehash.phash(img))
                    if img_hash not in image_hashes:
                        image_hashes[img_hash] = os.path.basename(file_path)  # Store only the filename
                except Exception as e:
                    print(f"Failed to process image {file_path}: {str(e)}")
            else:
                print(f"Skipped non-image file: {os.path.join(subdir, file)}")
    return image_hashes

def rename_images_based_on_hashes(source_hashes, target_directory):
    """Rename images in the target directory based on source hashes."""
    print(f"Renaming images in {target_directory}...")
    for subdir, dirs, files in os.walk(target_directory):
        for file in files:
            file_path = os.path.join(subdir, file)  # Define file_path here for broad scope
            if file.lower().endswith(('.jpg', '.png', '.jpeg')) and not file.startswith('.'):
                try:
                    with Image.open(file_path) as img:
                        img_hash = str(imagehash.phash(img))
                    if img_hash in source_hashes:
                        new_filename = source_hashes[img_hash]
                        new_file_path = os.path.join(subdir, new_filename)
                        if new_file_path != file_path and not os.path.exists(new_file_path):
                            os.rename(file_path, new_file_path)
                            print(f"Renamed {file_path} to {new_file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
            else:
                print(f"Skipped non-image or unsupported file: {file_path}")

# Paths for directories
root_directory = 'biases'
age_directory = os.path.join(root_directory, 'age')
gender_directory = os.path.join(root_directory, 'gender')
reference_directory = '../dataset'

# Hash images in the reference directory
reference_image_hashes = hash_images(reference_directory)

# Rename images in both 'age' and 'gender' directories based on reference hashes
rename_images_based_on_hashes(reference_image_hashes, age_directory)
rename_images_based_on_hashes(reference_image_hashes, gender_directory)