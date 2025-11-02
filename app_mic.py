import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib, json, os, tempfile
from streamlit_mic_recorder import mic_recorder
from sklearn.preprocessing import StandardScaler

# === Konfigurasi ===
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
    # pastikan selalu mono, 16kHz, 1 detik
    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

    # hilangkan bagian hening
    y, _ = librosa.effects.trim(y, top_db=30)

    # normalisasi panjang
    if len(y) < FIX_SAMPLES:
        y = np.pad(y, (0, FIX_SAMPLES - len(y)))
    else:
        y = y[:FIX_SAMPLES]

    # normalisasi amplitudo
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # ekstraksi fitur
    feats = extract_features(y, sr).reshape(1, -1)
    feats_scaled = scaler.transform(feats)

    # prediksi
    probs = model.predict_proba(feats_scaled)[0]
    pred = model.predict(feats_scaled)[0]

    label = classes[int(pred)]
    conf = float(np.max(probs))
    return label, conf, dict(zip(classes, probs.tolist()))


# === UI Streamlit ===
st.set_page_config(page_title="🎤 Voice Command Detector (Mic)", layout="centered")
st.title("🎙️ Deteksi Suara 'Buka' / 'Tutup'")
st.markdown("Tekan tombol di bawah untuk merekam suara langsung dari mikrofon.")

# Load model
model, scaler, classes = load_assets()

# === Rekam suara ===
audio_data = mic_recorder(
    start_prompt="🎙️ Tekan untuk mulai merekam",
    stop_prompt="🛑 Tekan lagi untuk berhenti",
    key="recorder",
    just_once=False
)

if audio_data:
    # Simpan hasil rekaman ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_data["bytes"])
        tmp_path = tmp.name

    st.audio(tmp_path, format="audio/wav")
    st.success("✅ Suara berhasil direkam!")

    if st.button("🔍 Prediksi Sekarang"):
        label, conf, probs = predict_audio(tmp_path, model, scaler, classes)

        st.success(f"**Prediksi:** {label.upper()}  \n**Kepercayaan:** {conf*100:.2f}%")
        st.json(probs)

        # Tambahkan threshold kepercayaan
        if conf < 0.7:
            st.warning("⚠️ Suara tidak dikenali dengan cukup yakin. Coba ulangi rekaman.")
        else:
            if label.lower() == "buka":
                st.markdown("🟢 Sistem mengenali suara **BUKA**.")
            elif label.lower() == "tutup":
                st.markdown("🔴 Sistem mengenali suara **TUTUP**.")

