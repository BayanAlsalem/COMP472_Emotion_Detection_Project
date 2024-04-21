# COMP 472 - Emotion Detection Project

## Overview
This project is a part of the COMP472 class requirement. It aims to predict a student's emotions using a CNN architecture.

## Contents
- `main.py`: Script to clean data, visualize, and train the model using the cleaned dataset.
- `evaluate.py`: Contains code to evaluate the best model's performance on a test set or to predict a single image's class.
- `cnn_4.py`: Defines our main machine learning model architecture.
- `cnn_var_1.py`: Defines our first variant from the main model.
- `cnn_var_2.py`: Defines our latest CNN architecture model for image classification.
- `best_model.pth`: The latest trained model for the `cnn_var_2` model with an accuracy of 63%.
- `best_model_confusion_matrix.png`: Confusion matrix of the trained best model.
- **`bias_preparation/README.md`**: Documentation for preparing and handling biases in the dataset, ensuring model robustness.

## Getting Started
### Prerequisites
List of libraries and tools that need to be installed:
- Python 3.8.19
- PyTorch 1.8.1
- Matplotlib 3.7.2
- Numpy 1.24.3

### Installation
Please follow the TA's instructions to setup and activate the Anaconda environment:
```
conda activate pytorch_env
```

## Usage

### Training the Model
Run `main.py` to clean data, visualize, and train the model:
```
python main.py
```

### Evaluating the Model
Update the following inside `evaluate.py` for test dataset evaluation:
- Set `data_dir` to the testset folder.
- Set `evaluateAllDataset` to True.

```
data_dir = "dataset"
model_dir = "best_model.pth"
evaluateAllDataset = True
```
Run the evaluate.py script:
```
python evaluate.py
```

Example output:
```
Accuracy on dataset: 62.95%
```

### Applying the Models
To predict on an image, update the following inside `evaluate.py`:
- Set `image_dir` to the test image.
- Set `evaluateAllDataset` to False.

```
model_dir = "best_model.pth"
image_dir = "dataset/happy/im62.png"
evaluateAllDataset = False
```
Run the evaluate.py script:
```
python evaluate.py
```

Example output:
```
Predicted class: happy
```
---

<!--## Contributing-->
<!--(Optional) Explain how others can contribute to your project. Include any guidelines they should follow.-->

<!--## License-->
<!--(Optional) State the license under which your project is released, allowing others to know how they can use it.-->

<!------->

<!--Feel free to customize this template based on the specifics of your project. The key is to make sure that anyone who wants to use your code can easily understand what each part does and how to run it.-->