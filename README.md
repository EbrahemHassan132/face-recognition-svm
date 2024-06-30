# Face Recognition with SVM

This project implements a face recognition system using Support Vector Machine (SVM) for classification. The system is capable of detecting faces, extracting facial embeddings using FaceNet, and classifying faces into predefined classes.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Contributing](#contributing)
- [License](#license)

## Features
- Face detection using MTCNN
- Face embedding extraction using FaceNet
- SVM for face classification
- Real-time face recognition using a webcam

## Requirements
- Python 3.6+
- OpenCV
- TensorFlow
- Keras
- MTCNN
- scikit-learn
- pickle
- matplotlib

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/EbrahemHassan132/face-recognition-svm.git
    ```
2. Navigate to the project directory:
    ```bash
    cd face-recognition-svm
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your dataset as described in the [Dataset Preparation](#dataset-preparation) section.
2. Run the training script to train the SVM model:
    ```bash
    python train.py
    ```
3. Run the real-time face recognition script:
    ```bash
    python recognize.py
    ```

## Project Structure
```
face-recognition-svm/
│
├── dataset/
│   ├── person1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── person2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
│
├── faces_embeddings_6classes.npz
├── haarcascade_frontalface_default.xml
├── svm_model_160x160_group.pkl
├── train.py
├── recognize.py
├── requirements.txt
└── README.md
```

## Dataset Preparation
1. Create a directory named `dataset` in the project root.
2. Inside the `dataset` directory, create subdirectories for each person you want to recognize.
3. Add face images to the corresponding subdirectories. Ensure that the images are clear and properly labeled.

## Training the Model
1. Run the `create_the_model.py` script to load the dataset, extract face embeddings, and train the SVM model.
2. The script will save the trained model as `svm_model_160x160_group.pkl` and the face embeddings as `faces_embeddings_6classes.npz`.

## Testing the Model
1. Run the `main.py` script to perform real-time face recognition using your webcam.
2. The script will display the recognized faces along with their names and confidence scores.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

## Acknowledgements
- [OpenCV](https://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [MTCNN](https://github.com/ipazc/mtcnn)
- [FaceNet](https://github.com/nyoki-mtl/keras-facenet)

---

Feel free to customize this README file to suit your project's specifics and include any additional information that might be relevant.