import streamlit as st
import numpy as np
import librosa
import librosa.display
import os
import json
from tensorflow.keras.models import load_model
from fpdf import FPDF
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="InstruNet AI", layout="wide")

# ==========================
# CREDENTIALS
# ==========================
CREDENTIALS = {
    "mohith": "mohith123",
    "friend": "friend123"
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
        mel_db = np.pad(mel_db, ((0,0),(0,MAX_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:,:MAX_FRAMES]
    mel_db = mel_db[np.newaxis, ..., np.newaxis]
    probs = model.predict(mel_db, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    results = {inv_label_map[i]: float(probs[i]) for i in top_indices}
    return results

# ==========================
# PDF REPORT FUNCTION
# ==========================
def generate_pdf_report(predictions, audio_name, figures=[]):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "InstruNet AI - Instrument Recognition Report", ln=True, align="C")

    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    pdf.cell(0,8, f"Audio File: {audio_name}", ln=True)
    pdf.cell(0,8, f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.ln(5)
    pdf.cell(0,8,"Detected Instruments:", ln=True)
    pdf.ln(3)
    for inst, conf in predictions.items():
        pdf.cell(80,8, inst, border=1)
        pdf.cell(40,8, f"{conf*100:.2f}%", border=1, ln=True)

    pdf.ln(5)
    pdf.cell(0,8,"Visualizations Included Below:", ln=True)

    for fig in figures:
        fig_path = "temp_fig.png"
        fig.savefig(fig_path, dpi=100)
        pdf.image(fig_path, w=180)
        os.remove(fig_path)

    return pdf.output(dest="S").encode("latin1")

# ==========================
# LOGIN PAGE
# ==========================
def login_page():
    st.markdown("""
        <h1 style='color:#4B0082;'>🎵 InstruNet AI</h1>
        <h3 style='color:#6A5ACD;'>CNN-Based Music Instrument Recognition System</h3>
    """, unsafe_allow_html=True)

    st.info("Welcome! Please login to continue.")

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
    st.markdown(f"<h1 style='color:#4B0082;'>🎵 InstruNet AI</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:#6A5ACD;'>CNN-Based Music Instrument Recognition System</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])
    predictions = None
    waveform_fig, mel_fig = None, None

    if uploaded_file is not None:
        temp_path = "temp.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(temp_path)

        predictions = predict_instruments(temp_path)

        if predictions:
            st.subheader("🎯 Top Predicted Instruments")
            for inst, conf in predictions.items():
                st.write(f"**{inst}** : {conf*100:.2f}%")
                st.progress(int(conf*100))

        os.remove(temp_path)

    # Visualization button
    if predictions and st.button("📊 Visualization"):
        y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE, mono=True)

        st.subheader("🌊 Waveform")
        waveform_fig, ax = plt.subplots(figsize=(12,3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(waveform_fig)

        st.subheader("🎨 Mel Spectrogram")
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_fig, ax = plt.subplots(figsize=(12,4))
        img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
        mel_fig.colorbar(img, ax=ax, format='%+2.0f dB')
        st.pyplot(mel_fig)

        st.subheader("📈 Instrument Intensity Timeline")
        timeline_fig, ax = plt.subplots(figsize=(12,3))
        instruments = list(predictions.keys())
        confidences = list(predictions.values())
        ax.bar(instruments, confidences, color='purple')
        ax.set_ylabel('Confidence')
        timeline_fig.tight_layout()
        st.pyplot(timeline_fig)

        # Export buttons
        st.subheader("📤 Export")
        json_data = json.dumps(predictions, indent=4)
        st.download_button("Download JSON", data=json_data, file_name="instrument_report.json", mime="application/json")

        pdf_bytes = generate_pdf_report(predictions, uploaded_file.name, [waveform_fig, mel_fig, timeline_fig])
        st.download_button("Download PDF", data=pdf_bytes, file_name="instrument_report.pdf", mime="application/pdf")

# ==========================
# ENTRY POINT
# ==========================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_page()
else:
    main_app()