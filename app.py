from flask import Flask, request, send_file, jsonify
import cv2 as cv
import numpy as np
import pickle
from keras_facenet import FaceNet
import os
import random

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__, static_folder="static")
app.config["DATASET_PATH"] = (
    r"D:\Lecturs\Mechatronics Systems 2\face_recognition_app\dataset"  # Adjust this path as needed
)

# Load models and encoder at startup
facenet = FaceNet()
model = pickle.load(open("svm_model_160x160_group.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/dataset/<person_name>/<image_name>")
def serve_dataset_image(person_name, image_name):
    image_path = os.path.join(app.config["DATASET_PATH"], person_name, image_name)
    return send_file(image_path, mimetype="image/jpeg")


@app.route("/recognize", methods=["POST"])
def recognize():
    # Receive and decode the uploaded image
    file = request.files["image"]
    img = cv.imdecode(np.frombuffer(file.read(), np.uint8), cv.IMREAD_COLOR)
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"message": "No face detected"})
    else:
        # Process the first detected face
        x, y, w, h = faces[0]
        face_img = img[y : y + h, x : x + w]
        face_img = cv.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)

        # Generate embedding and predict
        embedding = facenet.embeddings(face_img)
        predictions = model.predict_proba(embedding)
        max_confidence_index = np.argmax(predictions)
        max_confidence = predictions[0][max_confidence_index]

        if max_confidence < 0.5:
            return jsonify({"message": "Unknown person"})
        else:
            final_name = encoder.inverse_transform([max_confidence_index])[0]
            person_dir = os.path.join(app.config["DATASET_PATH"], final_name)
            if os.path.isdir(person_dir):
                images = [
                    f
                    for f in os.listdir(person_dir)
                    if f.endswith((".jpg", ".jpeg", ".png"))
                ]
                selected_images = random.sample(images, min(5, len(images)))
                image_urls = [f"/dataset/{final_name}/{img}" for img in selected_images]
            else:
                image_urls = []
            return jsonify({"name": final_name, "images": image_urls})


if __name__ == "__main__":
    app.run(debug=True)
