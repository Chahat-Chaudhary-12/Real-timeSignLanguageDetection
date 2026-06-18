# Real-Time Sign Language Detection System

This project captures live webcam frames, detects a hand using MediaPipe, converts the hand pose into landmark features, and classifies the gesture into text with a TensorFlow model.

## Features

- Real-time webcam-based sign language recognition
- Hand landmark extraction with MediaPipe
- Dataset collection script for custom gestures
- Local TensorFlow training pipeline
- Live prediction smoothing for more stable output

## Project Structure

app.py - real-time webcam inference using MediaPipe + trained model, converts hand gestures into text and speech
collect_landmarks.py - captures hand landmark data using MediaPipe and saves them as .npy files for training
train_model.py - trains a deep learning model on extracted hand landmark data and saves the trained model (.keras) and label mapping (.pkl)
landmark_model.keras - trained Keras model used for gesture prediction
labels.pkl - label mapping file for converting model outputs into readable gestures
landmark_data/ - dataset directory containing saved landmark samples for each gesture class
artifacts/ (optional if you used it elsewhere) - stores trained model files and supporting outputs

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Collect Gesture Data

Capture samples for each sign you want the model to learn.

```bash
python collect_data.py --label A --samples 200
python collect_data.py --label B --samples 200
python collect_data.py --label HELLO --samples 200
```

Controls:

- Press `C` to start or pause sample capture
- Press `Q` to quit

## Step 2: Train the Model

```bash
python train_model.py --epochs 30 --batch-size 16
```

The trained model is saved to `artifacts/sign_language_model.keras` and labels are saved to `artifacts/labels.npy`.

## Step 3: Run Real-Time Detection

```bash
python app.py --threshold 0.70 --buffer-size 5
```

Controls:

- Press `Q` to quit the detection window

## How It Works

1. The webcam captures live video frames.
2. MediaPipe detects the hand and extracts 21 hand landmarks.
3. Landmarks are normalized so the model focuses on hand shape and relative position.
4. A TensorFlow neural network predicts the gesture class.
5. The predicted label is displayed on the OpenCV video window in real time.

## Notes

- You need balanced samples for every gesture to get reliable predictions.
- Good lighting and a simple background improve detection quality.
- The current system focuses on single-hand gestures. You can extend it for words, commands, or two-hand signs later.
