# ==========================
# STREAMLIT PAGE CONFIG
# ==========================
import streamlit as st
st.set_page_config(page_title="InstruNet AI", layout="wide")

# ==========================
# IMPORTS
# ==========================
import numpy as np
import librosa
import librosa.display
import os
import json
from tensorflow.keras.models import load_model
from fpdf import FPDF
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
import tempfile

# ==========================
# LOGIN CREDENTIALS
# ==========================
import os
USER_CREDENTIALS = {
    "admin": os.environ.get("INSTRU_NET_ADMIN_PASS", "12345"),
    "user1": os.environ.get("INSTRU_NET_USER1_PASS", "password1")
}
# ==========================
# CONFIG
# ==========================
MODEL_PATH = "cnn_music_instruments.h5"
LABEL_MAP_PATH = "label_map.json"

SAMPLE_RATE = 22050
DURATION = 2.5
N_MELS = 64
MAX_FRAMES = 87
TOP_K = 3

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource(show_spinner=True)
def load_cnn_model():
    return load_model(MODEL_PATH, compile=False)

try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ==========================
# LOAD LABEL MAP
# ==========================
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_instruments(file_path, top_k=TOP_K):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(audio) < DURATION * sr:
        audio = np.pad(audio, (0, int(DURATION * sr) - len(audio)))
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel)
    if mel_db.shape[1] < MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_FRAMES]
    mel_db = mel_db[np.newaxis, ..., np.newaxis]
    probs = model.predict(mel_db, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    results = {inv_label_map[i]: float(probs[i]) for i in top_indices}
    return results

# ==========================
# PDF REPORT FUNCTION
# ==========================
def generate_pdf_report(predictions, audio_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 10, "InstruNet AI Instrument Recognition Report", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Audio File: {audio_name}", ln=True)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Detected Instruments:", ln=True)
    pdf.ln(3)
    pdf.set_font("Arial", "", 12)
    for inst, conf in predictions.items():
        pdf.cell(80, 8, inst, border=1)
        pdf.cell(40, 8, f"{conf*100:.2f}%", border=1, ln=True)

    return pdf.output(dest="S").encode("latin1")

# ==========================
# SESSION STATE
# ==========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

# ==========================
# LOGIN PAGE
# ==========================
def login_page():
    st.title("🎵 InstruNet AI")
    st.subheader("CNN-Based Music Instrument Recognition System")
    st.markdown("---")
    st.write("Welcome! Please log in to continue.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username in CREDENTIALS and CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.user = username
                st.success(f"Welcome {username}!")
            else:
                st.error("❌ Incorrect username or password")

# ==========================
# MAIN APP
# ==========================
def main_app():
    st.title("🎵 InstruNet AI")
    st.subheader("CNN-Based Music Instrument Recognition System")
    st.write(f"Hello, **{st.session_state.user}**! Upload an audio file to start analysis.")
    
    uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])
    if uploaded_file is not None:
        temp_path = "temp.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(temp_path)
        st.session_state.audio_path = temp_path
        
        if st.button("Predict Instruments"):
            try:
                st.session_state.predictions = predict_instruments(temp_path)
                st.success("✅ Prediction complete!")
                for inst, conf in st.session_state.predictions.items():
                    st.write(f"**{inst}** : {conf*100:.2f}%")
                    st.progress(int(conf*100))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        
        if st.session_state.predictions:
            if st.button("📊 Visualization"):
                audio, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
                st.subheader("Waveform")
                fig_wav, ax = plt.subplots(figsize=(12, 3))
                librosa.display.waveshow(audio, sr=sr, ax=ax, color="#1f77b4")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                st.pyplot(fig_wav)
                
                st.subheader("Mel Spectrogram")
                mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                fig_mel, ax = plt.subplots(figsize=(12, 4))
                img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="viridis")
                fig_mel.colorbar(img, ax=ax, format="%+2.0f dB")
                st.pyplot(fig_mel)
                
                st.subheader("Instrument Intensity Timeline (Top Instruments)")
                times = np.linspace(0, len(audio)/sr, len(audio))
                fig_timeline, ax = plt.subplots(figsize=(12, 3))
                for inst, conf in st.session_state.predictions.items():
                    ax.plot(times, np.ones_like(times)*conf, label=inst)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Confidence")
                ax.legend()
                st.pyplot(fig_timeline)
                
                st.subheader("Export Results")
                st.download_button(
                    "📄 Download JSON",
                    json.dumps(st.session_state.predictions, indent=4),
                    file_name="instrument_report.json",
                    mime="application/json"
                )
                pdf_bytes = generate_pdf_report(st.session_state.predictions, uploaded_file.name)
                st.download_button(
                    "📄 Download PDF",
                    pdf_bytes,
                    file_name="instrument_report.pdf",
                    mime="application/pdf"
                )
        os.remove(temp_path)

# ==========================
# ENTRY POINT
# ==========================
if not st.session_state.authenticated:
    login_page()
else:
    main_app()
