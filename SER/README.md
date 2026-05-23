# Speech Emotion Recognition System

A machine learning based Speech Emotion Recognotion (SER) system that classifies emotions from speech audio using MFCC features.

# Features:

- Speech emotion recognition
- MFCC feature extraction
- Audio augmentation
- Traditional machine learning algorithms
- emotion prediction

# Dataset:

- RAVDESS Emotional Speech Audio Dataset
    Link -> https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

Supported emotions:
 - Neutral
 - Calm
 - Happy
 - Sad
 - Angry
 - Fearful
 - Disgust
 - Surprised

# Model Architecture

## Pipeline:

- Audio Input
- MFCC Extraction
- Data Augmentation
- Machine Learning Algorithm
- Emotion Prediction

### Feature Engineering:

- 40 MFCC coefficients
- Audion normalization

### Audio Augmentation

- Noise Injection
- Time stretching
- Data variation techniques

### Running the Project

```bash
cd SER
python app.py
```

1. Run above code in terminal
2. Visit the local gradio url
3. Upload an audio in left side audio upload / dropdown and click submit