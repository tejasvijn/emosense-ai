from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import numpy as np
import cv2
import base64
import time

from PIL import Image
from io import BytesIO

# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)
CORS(app)

# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model("../model/emotion_model.keras")

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# ==========================================
# FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# HEALTH ROUTE
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })

# ==========================================
# DETECT EMOTION
# ==========================================

@app.route("/api/detect", methods=["POST"])
def detect_emotion():

    start_time = time.time()

    try:

        data = request.get_json()

        if "image" not in data:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400

        image_data = data["image"]

        # Remove base64 prefix
        image_data = image_data.split(",")[1]

        # Decode image
        image_bytes = base64.b64decode(image_data)

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        frame = np.array(image)

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # No faces found
        if len(faces) == 0:
            return jsonify({
                "success": True,
                "faces": 0
            })

        results = []

        img_height, img_width = frame.shape[:2]

        # Process all faces
        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (48, 48))

            face = face.astype("float32") / 255.0

            face = np.expand_dims(face, axis=-1)

            face = np.expand_dims(face, axis=0)

            predictions = model.predict(
                face,
                verbose=0
            )[0]

            dominant_index = int(
                np.argmax(predictions)
            )

            dominant_emotion = EMOTIONS[
                dominant_index
            ]

            confidence = float(
                predictions[dominant_index]
            )

            probabilities = {}

            for i, emotion in enumerate(EMOTIONS):
                probabilities[emotion] = float(
                    predictions[i]
                )

            results.append({
                "dominant": dominant_emotion,
                "confidence": confidence,
                "probabilities": probabilities,
                "bbox_norm": [
                    x / img_width,
                    y / img_height,
                    w / img_width,
                    h / img_height
                ]
            })

        latency = int(
            (time.time() - start_time) * 1000
        )

        main_face = results[0]

        return jsonify({
            "success": True,
            "faces": len(results),
            "dominant": main_face["dominant"],
            "confidence": main_face["confidence"],
            "probabilities": main_face["probabilities"],
            "all_faces": results,
            "latency_ms": latency
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
    