import os


def rename_images(folder_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter out non-image files
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]

    # Initialize count for renaming
    count = 822

    # Iterate over image files and rename them
    for file in image_files:
        # Check if file is already named in the format "img+number"
        if file.lower().startswith("img") and file[3:].isdigit():
            print(f"Skipping {file} as it is already named.")
        else:
            # Increment count even if we skip renaming
            count += 1

            # Generate new filename
            new_filename = f"img{count}.{file.split('.')[-1]}"

            # Check if the new filename already exists
            while os.path.exists(os.path.join(folder_path, new_filename)):
                # If so, increment count and generate a new filename
                count += 1
                new_filename = f"img{count}.{file.split('.')[-1]}"

            # Rename the file
            os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_filename))
            print(f"Renamed {file} to {new_filename}")


# Provide the folder path where the images are located
folder_path = "./focused"
rename_images(folder_path)
