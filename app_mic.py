import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import joblib, json, os, tempfile
from streamlit_mic_recorder import mic_recorder
from sklearn.preprocessing import StandardScaler
import soundfile as sf
from pydub import AudioSegment
import io


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
    try:
        # Gunakan soundfile (lebih stabil di Streamlit Cloud)
        y, sr = sf.read(file_path, dtype='float32')
    except Exception as e:
        raise RuntimeError(f"Gagal membaca file audio dengan soundfile: {e}")

    # Jika audio stereo, ubah ke mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Hilangkan keheningan (pakai librosa tapi hanya untuk trimming array, bukan loading file)
    try:
        y, _ = librosa.effects.trim(y, top_db=30)
    except Exception:
        pass

    # Normalisasi panjang (1 detik)
    FIX_SAMPLES = int(16000 * 1.0)
    if len(y) < FIX_SAMPLES:
        y = np.pad(y, (0, FIX_SAMPLES - len(y)))
    else:
        y = y[:FIX_SAMPLES]

    # Normalisasi amplitudo
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # Ekstraksi fitur
    feats = extract_features(y, sr).reshape(1, -1)
    feats_scaled = scaler.transform(feats)

    # Prediksi
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
    # Konversi bytes dari mic ke format WAV 16kHz agar soundfile bisa membaca
    audio_bytes = audio_data["bytes"]

    # Pastikan format input adalah WAV (mic_recorder bisa hasilkan webm)
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
    except Exception:
        # fallback jika format bukan WAV
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")

    # Konversi ke mono 16kHz PCM
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    # Simpan hasil konversi ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio.export(tmp, format="wav")
        tmp_path = tmp.name

    st.audio(tmp_path, format="audio/wav")
    st.success("✅ Suara berhasil direkam dan dikonversi!")

    if st.button("🔍 Prediksi Sekarang"):
        label, conf, probs = predict_audio(tmp_path, model, scaler, classes)
        st.success(f"**Prediksi:** {label.upper()}  \n**Kepercayaan:** {conf*100:.2f}%")
        st.json(probs)

        # Threshold
        if conf < 0.7:
            st.warning("⚠️ Suara tidak dikenali dengan cukup yakin. Coba ulangi rekaman.")
        else:
            if label.lower() == "buka":
                st.markdown("🟢 Sistem mengenali suara **BUKA**.")
            elif label.lower() == "tutup":
                st.markdown("🔴 Sistem mengenali suara **TUTUP**.")

