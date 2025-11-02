import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib, json, os
from tempfile import NamedTemporaryFile

# === Konfigurasi dasar ===
MODEL_PATH = "model/voice_cmd_best.pkl"
SCALER_PATH = "model/scaler.pkl"
CLASSES_PATH = "model/classes.json"
TARGET_SR = 16000
FIX_SECONDS = 1.0
FIX_SAMPLES = int(TARGET_SR * FIX_SECONDS)

# === Fungsi bantu ===
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    classes = json.load(open(CLASSES_PATH))
    return model, scaler, classes

def load_audio_fixed(path, target_sr=TARGET_SR, fix_len=FIX_SAMPLES):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    if len(y) < fix_len:
        y = np.pad(y, (0, fix_len - len(y)))
    else:
        y = y[:fix_len]
    return y, sr

def extract_features(y, sr=TARGET_SR, n_mfcc=20, n_fft=512, hop_length=160, win_length=400):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                hop_length=hop_length, win_length=win_length)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat = np.concatenate([mfcc, d1, d2], axis=0)
    mean = np.mean(feat, axis=1)
    std = np.std(feat, axis=1)
    return np.hstack([mean, std]).astype(np.float32)

def predict_audio(file_path, model, scaler, classes):
    ysig, _ = load_audio_fixed(file_path)
    feats = extract_features(ysig).reshape(1, -1)
    try:
        feats_scaled = scaler.transform(feats)
        probs = model.predict_proba(feats_scaled)[0]
        pred = model.predict(feats_scaled)[0]
    except Exception:
        probs = model.predict_proba(feats)[0]
        pred = model.predict(feats)[0]
    label = classes[int(pred)]
    conf = float(np.max(probs))
    return label, conf, dict(zip(classes, probs.tolist()))

# === UI Streamlit ===
st.set_page_config(page_title="🎤 Voice Command Detector", layout="centered")
st.title("🎙️ Prediksi Suara 'Buka' / 'Tutup'")
st.markdown("Upload file `.wav` hasil rekamanmu (durasi ~1 detik).")

model, scaler, classes = load_assets()

uploaded = st.file_uploader("Unggah file suara (.wav)", type=["wav"])
if uploaded is not None:
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.audio(tmp_path, format="audio/wav")

    if st.button("🔍 Prediksi"):
        label, conf, probs = predict_audio(tmp_path, model, scaler, classes)
        st.success(f"**Prediksi:** {label.upper()}  \n**Kepercayaan:** {conf*100:.2f}%")
        st.json(probs)

        if label.lower() == "buka":
            st.markdown("🟢 Sistem mengenali suara **BUKA**.")
        elif label.lower() == "tutup":
            st.markdown("🔴 Sistem mengenali suara **TUTUP**.")
