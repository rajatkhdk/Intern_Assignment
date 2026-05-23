import joblib
import gradio as gr
import librosa
import numpy as np
from scipy.stats import kurtosis, skew
from utility import extract_features, EMOTION_MAP

# Load the trained model
model = joblib.load('ser_model.pkl')



def predict_emotion(audio):
    y, sr = librosa.load(audio, sr=22050, duration=7, offset=3)
    features = extract_features(y, sr)
    features = features.reshape(1, -1)  # Reshape for prediction
    emotion = model.predict(features)[0]
    return EMOTION_MAP[emotion]
    # return emotion

# Gradio Interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Speech Emotion Recognition",
    description="Upload an audio file to predict the emotion expressed in the speech."
)
interface.launch()

