import librosa
import numpy as np
from scipy.stats import kurtosis, skew

# SHARED FEATURE EXTRACTOR (USE IN BOTH TRAINING + GRADIO)
def extract_features(y, sr):

  # ---------------- MFCC -----------------

  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
  mfcc_mean = np.mean(mfcc, axis=1)
  mfcc_std  = np.std(mfcc, axis=1)
  mfcc_kurtosis = kurtosis(mfcc, axis=1)
  mfcc_skew = skew(mfcc, axis=1)
  mfcc_max = np.max(mfcc, axis=1)
  mfcc_min = np.min(mfcc, axis=1)

  # ---------------- PITCH (F0) ----------------
  pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

  pitch = pitches[pitches > 0]

  if len(pitch) > 0:
      pitch_mean = np.mean(pitch)
      pitch_std = np.std(pitch)
      pitch_max = np.max(pitch)
      pitch_min = np.min(pitch)
  else:
      pitch_mean = pitch_std = pitch_max = pitch_min = 0

  # ---------------- Delta ----------------

  delta1 = librosa.feature.delta(mfcc, order=1)
  delta1_mean = np.mean(delta1, axis=1)
  delta1_std  = np.std(delta1, axis=1)
  delta1_kurtosis = kurtosis(delta1, axis=1)
  delta1_skew = skew(delta1, axis=1)
  delta1_max = np.max(delta1, axis=1)
  delta1_min = np.min(delta1, axis=1)

  delta2 = librosa.feature.delta(mfcc, order=2)
  delta2_mean = np.mean(delta2, axis=1)
  delta2_std  = np.std(delta2, axis=1)
  delta2_kurtosis = kurtosis(delta2, axis=1)
  delta2_skew = skew(delta2, axis=1)
  delta2_max = np.max(delta2, axis=1)
  delta2_min = np.min(delta2, axis=1)

  # ---------------- Mel ----------------
  mel = librosa.feature.melspectrogram(y=y, sr=sr)
  mel_mean = np.mean(mel, axis=1)
  mel_std  = np.std(mel, axis=1)
  mel_kurtosis = kurtosis(mel, axis=1)
  mel_skew = skew(mel, axis=1)
  mel_max = np.max(mel, axis=1)
  mel_min = np.min(mel, axis=1)

  # ---------------- Spectral Contrast ----------------
  contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
  contrast_mean = np.mean(contrast, axis=1)
  contrast_std  = np.std(contrast, axis=1)
  contrast_kurtosis = kurtosis(contrast, axis=1)
  contrast_skew = skew(contrast, axis=1)
  contrast_max = np.max(contrast, axis=1)
  contrast_min = np.min(contrast, axis=1)

  # ---------------- ZCR ----------------
  zcr = librosa.feature.zero_crossing_rate(y)
  zcr_mean = np.mean(zcr)
  zcr_std  = np.std(zcr)
  zcr_kurtosis = kurtosis(zcr.flatten())
  zcr_skew = skew(zcr.flatten())
  zcr_max = np.max(zcr)
  zcr_min = np.min(zcr)

  # ---------------- RMS ----------------
  rms = librosa.feature.rms(y=y)
  rms_mean = np.mean(rms)
  rms_std  = np.std(rms)
  rms_kurtosis = kurtosis(rms.flatten())
  rms_skew = skew(rms.flatten())
  rms_max = np.max(rms)
  rms_min = np.min(rms)

  # ---------------- FINAL VECTOR ----------------
  features = np.hstack([
      mfcc_mean, mfcc_std, mfcc_kurtosis, mfcc_skew, mfcc_max, mfcc_min,
      pitch_mean, pitch_std, pitch_max, pitch_min,
      delta1_mean, delta1_std, delta1_kurtosis, delta1_skew, delta1_max, delta1_min,
      delta2_mean, delta2_std, delta2_kurtosis, delta2_skew, delta2_max, delta2_min,
      mel_mean, mel_std, mel_kurtosis, mel_skew, mel_max, mel_min,
      contrast_mean, contrast_std, contrast_kurtosis, contrast_skew, contrast_max, contrast_min,
      zcr_mean, zcr_std, zcr_kurtosis, zcr_skew, zcr_max, zcr_min,
      rms_mean, rms_std, rms_kurtosis, rms_skew, rms_max, rms_min
  ])

  return features

EMOTION_MAP = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised",
}