# EmoSense AI — Real-Time Emotion Detection System

A real-time facial emotion recognition system that detects 7 human emotions through live webcam input, with dynamic UI visualization and confidence scoring.

## Features
- Custom CNN model trained from scratch on Google Colab
- Real-time facial emotion detection using OpenCV
- Flask backend serving a custom HTML/CSS/JS frontend
- Confidence scoring with dynamic UI visualization
- Frame-sampling optimization for smooth real-time inference
- Full offline capability

## Tech Stack
- **Model:** TensorFlow / Keras (CNN trained on Google Colab)
- **Backend:** Python, Flask, OpenCV
- **Frontend:** HTML, CSS, JavaScript

## Project Structure
  emosense-ai/

├── backend/

│   └── app.py

├── frontend/

│   └── index.html

└── model/

└── emotion_model.keras

## How to Run

1. Clone the repository

git clone https://github.com/tejasvijn/emosense-ai.git

cd emosense-ai

2. Install dependencies

pip install flask opencv-python tensorflow numpy

3. Run the backend

cd backend

python app.py

4. Open frontend/index.html in your browser (or visit the local server address shown in the terminal)

## Author

**Tejasvi J N**
[GitHub](https://github.com/tejasvijn)
