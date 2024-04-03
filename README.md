# COMP 472 - Emotion Detection Project

## Overview
This project is a part of COMP472 class requirement.

This model predict student's emotion based using a CNN architecture.

## Contents
- `main.py`: Script to data clean, visualize, and train the model using the cleaned dataset.

- `evaluate.py`: Contains code to evaluate the best model's performance on a test set or predict single image's class.
<br/>

- `cnn_4.py`: Defines our main machine learning model architecture (convolutional neural network for image classification).
- `cnn_var_1.py`: Defines our first variant from the main model.
- `cnn_var_2.py`: Defines our latest cnn architecture model for image classification.
<br/>

- `best_model.pth`: Latest trained best model for cnn_var_2 model with Accuracy of 63%.

- `best_model_confusion_matrix.png`: Confusion Matrix of the trained best model.
<br/>
- `README.md`: This file, describing the project and how to run the scripts.

## Getting Started
### Prerequisites
List any libraries or tools that need to be installed before running your project, including specific versions if necessary. For example:
- Python 3.8.19
- PyTorch 1.8.1
- Matplotlib 3.7.2
- Numpy 1.24.3

### Installation
Please follow the TA's instructions to setup and activate Anaconda environment:
```
conda activate pytorch_env
```

## Usage

### Training the Model
1. Running main.py will data clean, visualize, and train the model consecutively:
```
python main.py
```

### Evaluating the Model
1. To evaluate the trained model on the test dataset (Update the following inside evaluate.py):
    1. Update data_dir variable to be the testset folder.
    2. Update evaluateAllDataset to True
```
data_dir = "dataset"
model_dir = "best_model.pth"
evaluateAllDataset = True
```
2. Run the evaluate.py python script.
```
python evaluate.py
```

2. Evaluation metrics will be printed out, showing the model's performance.
<br/>

Example output:
```
(pytorch_env) moemenw@Moemens-MacBook-Pro COMP472_Emotion_Detection_Project % python evaluate.py

Accuracy on dataset: 62.95%
```

### Applying the Models
1. To apply the trained model and predict on an image (Update the following inside evaluate.py):
    1. Update image_dir variable to be the test image.
    2. Update evaluateAllDataset to False
```
model_dir = "best_model.pth"
image_dir = "dataset/happy/im62.png"
evaluateAllDataset = False
```
2. Run the evaluate.py python script.
```
python evaluate.py
```

2. Evaluation metrics will be printed out, showing the model's performance.
<br/>

Example output:
```
(pytorch_env) moemenw@Moemens-MacBook-Pro COMP472_Emotion_Detection_Project % python evaluate.py

Predicted class: happy
```
---

<!--## Contributing-->
<!--(Optional) Explain how others can contribute to your project. Include any guidelines they should follow.-->

<!--## License-->
<!--(Optional) State the license under which your project is released, allowing others to know how they can use it.-->

<!------->

<!--Feel free to customize this template based on the specifics of your project. The key is to make sure that anyone who wants to use your code can easily understand what each part does and how to run it.-->