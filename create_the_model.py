import cv2 as cv
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle

# Set TensorFlow log level to suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class FACELOADING:
    def __init__(self, directory):
        self.directory = directory
        self.target_size = (160, 160)
        self.X = []
        self.Y = []
        self.detector = MTCNN()

    def extract_face(self, filename):
        """
        Extracts and resizes the face from an image file.
        """
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x, y, w, h = self.detector.detect_faces(img)[0]["box"]
        x, y = abs(x), abs(y)
        face = img[y : y + h, x : x + w]
        face_arr = cv.resize(face, self.target_size)
        return face_arr

    def load_faces(self, dir):
        """
        Loads faces from a directory and returns a list of face images.
        """
        FACES = []
        for im_name in os.listdir(dir):
            try:
                path = dir + im_name
                single_face = self.extract_face(path)
                FACES.append(single_face)
            except Exception as e:
                pass
        return FACES

    def load_classes(self):
        """
        Loads all face images and their corresponding labels from the main directory.
        """
        for sub_dir in os.listdir(self.directory):
            path = self.directory + "/" + sub_dir + "/"
            FACES = self.load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"Loaded successfully: {len(labels)}")
            self.X.extend(FACES)
            self.Y.extend(labels)

        return np.asarray(self.X), np.asarray(self.Y)

    def plot_images(self):
        """
        Plots the loaded face images.
        """
        plt.figure(figsize=(18, 16))
        for num, image in enumerate(self.X):
            ncols = 3
            nrows = len(self.Y) // ncols + 1
            plt.subplot(nrows, ncols, num + 1)
            plt.imshow(image)
            plt.axis("off")


# Initialize the FACELOADING class with the dataset directory
faceloading = FACELOADING(r"D:\Lecturs\Mechatronics Systems 2\face_recognition\dataset")
X, Y = faceloading.load_classes()

# Plot the loaded face images
faceloading.plot_images()

# Initialize FaceNet model for face embeddings
embedder = FaceNet()


def get_embedding(face_img):
    """
    Generates an embedding for a given face image using FaceNet.
    """
    face_img = face_img.astype("float32")
    face_img = np.expand_dims(face_img, axis=0)
    yhat = embedder.embeddings(face_img)
    return yhat[0]


# Generate embeddings for all loaded face images
EMBEDDED_X = []
for img in X:
    EMBEDDED_X.append(get_embedding(img))
EMBEDDED_X = np.asarray(EMBEDDED_X)

# Save the embeddings and labels to a compressed file
np.savez_compressed("faces_embeddings_6classes.npz", EMBEDDED_X, Y)

# Encode the labels
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    EMBEDDED_X, Y, shuffle=True, random_state=17
)

# Train an SVM model on the training set
model = SVC(kernel="linear", probability=True)
model.fit(X_train, Y_train)

# Evaluate the model on the training and testing sets
ypreds_train = model.predict(X_train)
ypreds_test = model.predict(X_test)
accuracy_score(Y_train, ypreds_train)
accuracy_score(Y_test, ypreds_test)

# Detect faces in an example image for testing
detector = MTCNN()
results = detector.detect_faces(img)

# Load and preprocess the test image
t_im = cv.imread(
    r"D:\Lecturs\Mechatronics Systems 2\face_recognition\ebrahem_test.jpeg"
)
t_im = cv.cvtColor(t_im, cv.COLOR_BGR2RGB)
x, y, w, h = detector.detect_faces(t_im)[0]["box"]
t_im = t_im[y : y + h, x : x + w]
t_im = cv.resize(t_im, (160, 160))
test_im = get_embedding(t_im)

# Predict the class of the test image
test_im = [test_im]
ypreds = model.predict(test_im)
encoder.inverse_transform(ypreds)

# Save the trained SVM model to a file
with open("svm_model_160x160_group.pkl", "wb") as f:
    pickle.dump(model, f)

# plt.show()
