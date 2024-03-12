import os
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

# count function
def count_class_with_type(c, t):
    path = base_directory+'/'+t+"/"+c;
    if(os.path.exists(path)):
        return len([1 for x in list(os.scandir(path)) if x.is_file()])
    else: 
        return 0;

# Define the classes or labels and dataset types (train/test)
base_directory = 'dataset'
classes = ['happy', 'neutral', 'surprised', 'focused']
train_counts = [0 for c in classes];
test_counts = [0 for c in classes];

# Retrieve all counts
print('Class', 'Train', 'Test')
for i in range(len(classes)):
    train_counts[i] = count_class_with_type(classes[i], 'train')
    test_counts[i] = count_class_with_type(classes[i], 'test')


######################################################
###### STEP 1: SHOW ALL COUNT OF ALL CLASSEES ########
######################################################

# Defining the width of the bars (using the same width as before for consistency)
width = 0.35

# Plotting the bar graph
plt.figure(figsize=(12, 8))

# Defining the width of the bars (using the same width as before for consistency)
width = 0.35

bars1 = plt.bar(classes, train_counts, width, label='Training Data', color='skyblue')
bars2 = plt.bar(classes, test_counts, width, bottom=train_counts, label='Test Data', color='orange')

plt.xlabel('Class Names')
plt.ylabel('Number of Images')
plt.title('Class Distribution in Dataset: Combined Training and Test Data')
plt.legend()
plt.show()


######################################################
# STEP 2: SHOW SAMPLE 5X5 TRAIN DATA OF ALL CLASSEES #
######################################################

# Number of images and grid size
num_images = 25
grid_size = (5, 5)

for i in range(len(classes)):
    # Specify the path to your image directory
    image_dir_path = base_directory+'/train/'+classes[i]

    # List all files in the directory
    image_files = [os.path.join(image_dir_path, file) for file in os.listdir(image_dir_path) if file.endswith(('jpg', 'jpeg', 'png'))]

    # Randomly select 25 images
    selected_files = random.sample(image_files, 25)

    # Plotting the images in a 5x5 grid
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    fig.suptitle('25 Sample Images for `'+classes[i]+'` class')

    # Adjust layout
    plt.subplots_adjust(wspace=0, hspace=0)

    for ax, image_file in zip(axes.flatten(), selected_files):
        # Load the image from the file
        img = Image.open(image_file)

        # Convert to RGB if it has an alpha channel (TO MAINTAIN CONSISTENT COLOR CHANNELS)
        # if img.mode == 'RGBA':
        #     img = img.convert('RGB')

        # If the image data is float and in the range [0, 255], normalize to [0, 1]
        # img_data = np.array(img)
        # if img_data.dtype == np.float32 or img_data.dtype == np.float64:
        #     img_data = img_data / 255.0
                
        # Check if the image is grayscale and set the colormap accordingly
        if img.mode == 'L':  # 'L' indicates a grayscale image
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

        ax.axis('off')  # Hide axes
    plt.show()