import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib, json, os, tempfile, io
from streamlit_mic_recorder import mic_recorder
from sklearn.preprocessing import StandardScaler

# === KONFIGURASI DASAR ===
MODEL_PATH = "model/voice_cmd_best.pkl"
SCALER_PATH = "model/scaler.pkl"
CLASSES_PATH = "model/classes.json"
TARGET_SR = 16000
FIX_SECONDS = 1.0
FIX_SAMPLES = int(TARGET_SR * FIX_SECONDS)

# === LOAD MODEL DAN SCALER ===
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    classes = json.load(open(CLASSES_PATH))
    return model, scaler, classes

# === FUNGSI BANTU AUDIO ===
def extract_features(y, sr=TARGET_SR, n_mfcc=20, n_fft=512, hop_length=160, win_length=400):
    """Ekstraksi MFCC + delta + delta2 dengan pooling (mean + std)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft,
                                hop_length=hop_length, win_length=win_length)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat = np.concatenate([mfcc, d1, d2], axis=0)
    mean = np.mean(feat, axis=1)
    std = np.std(feat, axis=1)
    return np.hstack([mean, std]).astype(np.float32)

def predict_audio(file_path, model, scaler, classes):
    """Prediksi suara dari file .wav (aman tanpa audioread)."""
    try:
        y, sr = sf.read(file_path, dtype='float32')
    except Exception as e:
        raise RuntimeError(f"Gagal membaca file audio: {e}")

    # pastikan mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # hilangkan bagian hening
    try:
        y, _ = librosa.effects.trim(y, top_db=30)
    except Exception:
        pass

    # normalisasi panjang
    FIX_SAMPLES = int(TARGET_SR * FIX_SECONDS)
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

    # prediksi model
    probs = model.predict_proba(feats_scaled)[0]
    pred = model.predict(feats_scaled)[0]
    label = classes[int(pred)]
    conf = float(np.max(probs))
    return label, conf, dict(zip(classes, probs.tolist()))

# === ANTARMUKA STREAMLIT ===
st.set_page_config(page_title="🎤 Voice Command Detector", layout="centered")
st.title("🎙️ Deteksi Suara: 'Buka' / 'Tutup'")
st.markdown("Tekan tombol di bawah untuk merekam suara langsung dari mikrofon Anda.")

# === LOAD MODEL ===
model, scaler, classes = load_assets()

# === REKAMAN SUARA LANGSUNG ===
audio_data = mic_recorder(
    start_prompt="🎙️ Tekan untuk mulai merekam",
    stop_prompt="🛑 Tekan lagi untuk berhenti",
    key="recorder",
    just_once=False
)

if audio_data:
    audio_bytes = audio_data["bytes"]

    # coba baca langsung dengan soundfile
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        # jika format tidak dikenal (misalnya WebM), fallback manual
        import wave
        import struct
        audio_bytes_io = io.BytesIO(audio_bytes)
        data = np.frombuffer(audio_bytes_io.read(), dtype=np.int16).astype(np.float32) / 32768.0
        sr = TARGET_SR

    # ubah ke mono jika stereo
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # simpan sebagai file .wav valid
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp, data, sr, format="WAV", subtype="PCM_16")
        tmp_path = tmp.name

    st.audio(tmp_path, format="audio/wav")
    st.success("✅ Suara berhasil direkam dan dikonversi!")

    # tombol prediksi
    if st.button("🔍 Prediksi Sekarang"):
        label, conf, probs = predict_audio(tmp_path, model, scaler, classes)
        st.success(f"**Prediksi:** {label.upper()}  \n**Kepercayaan:** {conf*100:.2f}%")
        st.json(probs)

        # threshold kepercayaan
        if conf < 0.7:
            st.warning("⚠️ Suara tidak dikenali dengan cukup yakin. Coba ulangi rekaman.")
        else:
            if label.lower() == "buka":
                st.markdown("🟢 Sistem mengenali suara **BUKA**.")
            elif label.lower() == "tutup":
                st.markdown("🔴 Sistem mengenali suara **TUTUP**.")
