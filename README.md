# Traffic Sign Detection

## Overview
Traffic Sign Detection is a computer vision project focused on recognizing and classifying traffic signs from images. This system is designed for applications such as autonomous driving, advanced driver-assistance systems (ADAS), and road safety analytics. The goal is to accurately detect signs like speed limits, directional turns, and vehicle restrictions using image processing and machine learning techniques.

## Table of Contents
1. [Project Structure](#project-structure)  
2. [Installation](#installation)  
3. [Dataset](#dataset)  
4. [Usage](#usage)  
5. [Model Architecture](#model-architecture)  
6. [Results](#results)  
7. [Contributors](#contributors)  
8. [License](#license)

## Project Structure
```
Indian-Traffic-Sign-Detection-Classification/
│
├── data/
│   ├── 1.png          # Example: Left curve warning sign
│   ├── 2.jpg          # Example: Turn left mandatory sign
│   ├── 3.png          # Example: No trucks allowed sign
│   ├── 4.jpg          # Example: Speed limit 30 km/h sign
│   └── Test.csv       # Test dataset for evaluation
│
├── src/
│   └── model-training.ipynb  # Jupyter Notebook for training and testing the model
│
├── README.md          # Project documentation
└── requirements.txt   # Python dependencies
```

## Installation

### Requirements
- Python 3.x  
- Jupyter Notebook  
- OpenCV  
- TensorFlow or PyTorch  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn

Install the required dependencies with:
```bash
pip install -r requirements.txt
```

## Dataset
The project uses a collection of traffic sign images stored in the data/ directory. The dataset can be expanded by adding more labeled images in supported formats. Each image represents a specific traffic sign, including speed limits, directional instructions, and vehicle restrictions.

- **1.png**: Left curve warning
- **2.jpg**: Turn left mandatory
- **3.png**: No trucks allowed
- **4.jpg**: Speed limit 30 km/h
- **Test.csv**: Additional samples for testing and evaluation

## Usage

1. Clone the repository:
```bash
git clone https://github.com/Metal-Code/Indian-Traffic-Sign-Detection-Classification.git
cd Indian-Traffic-Sign-Detection-Classification
```

2. Open the `model-training.ipynb` notebook using Jupyter Notebook or Google Colab.

3. Run the notebook cells to preprocess images, train the model, and evaluate its performance.

4. To test new signs, add your images to the `data/` folder and update the dataset accordingly.

## Model Architecture
The traffic sign detection model uses a Convolutional Neural Network (CNN) with the following key components:

- **Input Layer**: Processes preprocessed traffic sign images.
- **Convolutional Layers**: Extract relevant features such as edges and shapes.
- **Pooling Layers**: Reduce the dimensionality of feature maps.
- **Fully Connected Layers**: Perform the final classification based on extracted features.

Details of the full architecture and training process are provided in the `model-training.ipynb` notebook.

## Results
After training, the model can accurately detect various traffic signs, including:

- Left curve warning
- Turn left mandatory
- No trucks allowed
- Speed limit 30 km/h

Performance metrics, sample predictions, and possible improvements are documented in the notebook. You can tune hyperparameters and expand the dataset to enhance accuracy.

## Contributors
- Add your name and contributions here

## License
This project is licensed under the MIT License - see the LICENSE file for details.
