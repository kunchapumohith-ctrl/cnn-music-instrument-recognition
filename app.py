# ==========================
# STREAMLIT PAGE CONFIG
# ==========================
import streamlit as st
st.set_page_config(page_title="InstruNet AI - Music Instrument Recognition", layout="wide")

# ==========================
# IMPORTS
# ==========================
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import json
from tensorflow.keras.models import load_model
from fpdf import FPDF
from datetime import datetime
from io import BytesIO

# ==========================
# LOGIN CREDENTIALS
# ==========================
CREDENTIALS = {
    "mohith": "mohith123",
    "mentor": "mentor123"
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
TOP_K = 3  # Show top 3 instruments

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
# SESSION STATE INITIALIZATION
# ==========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "visualizations" not in st.session_state:
    st.session_state.visualizations = {}

# ==========================
# LOGIN PAGE
# ==========================
def login_page():
    st.markdown("<h1 style='color:#4B0082'>Welcome to InstruNet AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#6A5ACD'>CNN-Based Music Instrument Recognition System</h3>", unsafe_allow_html=True)
    st.write("Please log in to continue:")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username in CREDENTIALS and CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.session_state.user = username
                st.success(f"✅ Welcome {username}!")
                st.experimental_rerun()
            else:
                st.error("❌ Invalid username or password")

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
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "InstruNet AI Instrument Recognition Report", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Audio File: {audio_name}", ln=True)
    pdf.cell(0, 8, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 8, "Top Predicted Instruments:", ln=True)
    pdf.ln(3)
    for inst, conf in predictions.items():
        pdf.cell(80, 8, inst, border=1)
        pdf.cell(40, 8, f"{conf*100:.2f}%", border=1, ln=True)
    pdf.ln(5)

    pdf.cell(0, 8, "Analysis Summary:", ln=True)
    pdf.cell(0, 8, f"Total Instruments Detected: {len(predictions)}", ln=True)
    pdf.cell(0, 8, f"Average Confidence: {np.mean(list(predictions.values()))*100:.2f}%", ln=True)
    pdf.ln(5)
    pdf.cell(0, 8, "Visualizations Included: Waveform, Mel Spectrogram, Intensity Timeline", ln=True)

    return pdf.output(dest="S").encode("latin1")

# ==========================
# MAIN APP
# ==========================
def main_app():
    st.markdown("<h1 style='color:#4B0082'>InstruNet AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#6A5ACD'>CNN-Based Music Instrument Recognition System</h3>", unsafe_allow_html=True)
    st.write(f"Logged in as: **{st.session_state.user}**")

    uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])
    if uploaded_file:
        temp_path = "temp.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(temp_path)

        try:
            st.session_state.predictions = predict_instruments(temp_path)
            st.session_state.audio_data = uploaded_file.name
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            os.remove(temp_path)
            return

        os.remove(temp_path)

        st.subheader("🎯 Top Predicted Instruments")
        for inst, conf in st.session_state.predictions.items():
            st.write(f"**{inst}** : {conf*100:.2f}%")
            st.progress(int(conf*100))

        if st.button("🔍 Visualization"):
            audio_bytes = uploaded_file.getvalue()
            y, sr = librosa.load(BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)

            st.markdown("### 🌊 Waveform")
            fig_wav, ax = plt.subplots(figsize=(12, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax, color='#4B0082')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig_wav)
            plt.close(fig_wav)

            st.markdown("### 🎨 Mel Spectrogram")
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            fig_mel, ax = plt.subplots(figsize=(12, 4))
            librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="viridis")
            fig_mel.colorbar(ax=ax, format="%+2.0f dB")
            st.pyplot(fig_mel)
            plt.close(fig_mel)

            st.markdown("### 📈 Instrument Intensity Timeline")
            fig_timeline, ax = plt.subplots(figsize=(12, 3))
            instruments = list(st.session_state.predictions.keys())
            intensities = list(st.session_state.predictions.values())
            ax.barh(instruments, intensities, color="#6A5ACD")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Confidence")
            st.pyplot(fig_timeline)
            plt.close(fig_timeline)

        # EXPORT SECTION
        st.subheader("📤 Export Reports")
        json_data = json.dumps(st.session_state.predictions, indent=4)
        st.download_button("📥 Download JSON", json_data, file_name="instrument_report.json", mime="application/json")
        pdf_bytes = generate_pdf_report(st.session_state.predictions, uploaded_file.name)
        st.download_button("📥 Download PDF", pdf_bytes, file_name="instrument_report.pdf", mime="application/pdf")

# ==========================
# ENTRY POINT
# ==========================
if not st.session_state.authenticated:
    login_page()
else:
    main_app()
