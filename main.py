import cv2 as cv
import numpy as np
import os

# import serial
# import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet


facenet = FaceNet()
faces_embeddings = np.load("faces_embeddings_6classes.npz")
Y = faces_embeddings["arr_1"]
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
model = pickle.load(open("svm_model_160x160_group.pkl", "rb"))


# ser = serial.Serial("COM9", 9600)

cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to retrieve frame from the camera.")
        break

    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
    for x, y, w, h in faces:
        img = rgb_img[y : y + h, x : x + w]
        img = cv.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)
        ypred = facenet.embeddings(img)
        predictions = model.predict_proba(ypred)
        class_indices = model.classes_
        max_confidence_index = np.argmax(predictions)
        max_confidence = predictions[0][max_confidence_index]
        final_name = encoder.inverse_transform([max_confidence_index])[0]
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

        # if (
        #     final_name in ["Ebrahem", "jenna_ortega", "robert_downey", "taylor_swift"]
        #     and max_confidence > 0.90
        # ):
        #     ser.write(b"1")
        #     time.sleep(5)
        #     ser.write(b"0")
        #     pass

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
