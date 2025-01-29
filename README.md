# FER-2013-Emotion-Recognition-Model

# README.md

# Emotion Recognition Using FER-2013

This repository contains a deep learning model for real-time emotion recognition using the FER-2013 dataset. The model is trained using a convolutional neural network (CNN) and can classify emotions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Features
- Trained on the FER-2013 dataset
- Real-time emotion detection using OpenCV
- Uses a Convolutional Neural Network (CNN) model built with TensorFlow/Keras

## Installation

1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/FER-2013-Emotion-Recognition.git
   cd FER-2013-Emotion-Recognition
   ```

2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Download the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place it in the `data/` directory.

## Training the Model
To train the model on the FER-2013 dataset, run:
```sh
python train_model.py
```

## Running the Emotion Detection Model
To run real-time emotion detection using your webcam, execute:
```sh
python detect_emotion.py
```

Press 'q' to exit the webcam interface.

## File Structure
```
FER-2013-Emotion-Recognition/
│-- data/                # Folder containing the FER-2013 dataset
│-- models/              # Folder to save trained models
│-- train_model.py       # Script to train the CNN model
│-- detect_emotion.py    # Script for real-time emotion detection
│-- requirements.txt     # Dependencies
│-- README.md            # Project documentation
```

## Dependencies
The project requires the following dependencies:
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

to install them, run:
```sh
pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn
```

## License
This project is open-source and available under the MIT License.
