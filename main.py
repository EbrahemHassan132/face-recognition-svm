import cv2 as cv
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet

# Set TensorFlow log level to suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Initialize FaceNet model for face embeddings
facenet = FaceNet()

# Load precomputed face embeddings and labels
faces_embeddings = np.load("faces_embeddings_6classes.npz")
Y = faces_embeddings["arr_1"]

# Encode labels
encoder = LabelEncoder()
encoder.fit(Y)

# Load Haar cascade for face detection
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load pre-trained SVM model for face recognition
model = pickle.load(open("svm_model_160x160_group.pkl", "rb"))

# Initialize webcam capture
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame from the camera.")
        break

    # Convert frame to RGB and grayscale for face detection and recognition
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    # Iterate over detected faces
    for x, y, w, h in faces:
        img = rgb_img[y : y + h, x : x + w]
        img = cv.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)

        # Get face embeddings using FaceNet
        ypred = facenet.embeddings(img)

        # Predict the face using SVM model
        predictions = model.predict_proba(ypred)
        class_indices = model.classes_

        # Get the class with the highest confidence
        max_confidence_index = np.argmax(predictions)
        max_confidence = predictions[0][max_confidence_index]
        final_name = encoder.inverse_transform([max_confidence_index])[0]

        # Draw rectangle around the face and display the name with confidence
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 10)
        text = f"{final_name}: {max_confidence:.2f}"
        cv.putText(
            frame,
            text,
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
            cv.LINE_AA,
        )

        # Optional: Send signal to serial port for specific recognized faces
        # if (
        #     final_name in ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5", "Class 6"]
        #     and max_confidence > 0.90
        # ):
        #     ser.write(b"1")
        #     time.sleep(5)
        #     ser.write(b"0")
        #     pass

    # Display the frame with face recognition
    cv.imshow("Face Recognition:", frame)

    # Exit the loop when 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
