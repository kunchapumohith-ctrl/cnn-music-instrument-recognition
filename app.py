# ==========================
# STREAMLIT APP - InstruNet AI
# ==========================
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from fpdf import FPDF
import json
import tempfile
import os
from datetime import datetime
from tensorflow.keras.models import load_model

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

# Login credentials
CREDENTIALS = {
    "mohith": "instrunet123",
    "user": "password123"
}

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
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
def predict_instruments(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(audio) < DURATION * sr:
        audio = np.pad(audio, (0, int(DURATION * sr) - len(audio)))

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel)
    if mel_db.shape[1] < MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0,0),(0,MAX_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_FRAMES]

    mel_db = mel_db[np.newaxis, ..., np.newaxis]
    probs = model.predict(mel_db, verbose=0)[0]

    top_indices = np.argsort(probs)[::-1][:TOP_K]
    results = {inv_label_map[i]: float(probs[i]) for i in top_indices}
    return results

# ==========================
# PDF REPORT FUNCTION
# ==========================
def generate_pdf(predictions, audio_name, waveform_fig, mel_fig, timeline_fig):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(0, 10, "InstruNet AI - Instrument Recognition Report", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(0,0,0)
    pdf.cell(0, 8, f"Audio File: {audio_name}", ln=True)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    pdf.cell(0, 8, "Detected Instruments:", ln=True)

    for inst, conf in predictions.items():
        pdf.cell(80, 8, inst, border=1)
        pdf.cell(40, 8, f"{conf*100:.2f}%", border=1, ln=True)

    # Save visualizations
    for i, fig in enumerate([waveform_fig, mel_fig, timeline_fig], start=1):
        if fig:
            temp_img = f"temp_{i}.png"
            fig.savefig(temp_img)
            pdf.add_page()
            pdf.image(temp_img, x=10, y=20, w=180)
            os.remove(temp_img)

    return pdf.output(dest="S").encode("latin1")

# ==========================
# LOGIN PAGE
# ==========================
def login_page():
    st.title("🎵 Welcome to InstruNet AI")
    st.subheader("CNN-Based Music Instrument Recognition System")
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
                st.error("Incorrect username or password!")

# ==========================
# MAIN APP
# ==========================
def main_app():
    st.title("🎵 InstruNet AI")
    st.subheader("CNN-Based Music Instrument Recognition System")

    uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])
    waveform_fig = mel_fig = timeline_fig = None
    predictions = None

    if uploaded_file:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.audio(temp_path)
        predictions = predict_instruments(temp_path)

        st.subheader("🎯 Top Predicted Instruments")
        for inst, conf in predictions.items():
            st.write(f"**{inst}** : {conf*100:.2f}%")
            st.progress(int(conf * 100))

        # ==========================
        # VISUALIZATION BUTTON
        # ==========================
        if st.button("📊 Show Visualizations"):
            y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
            
            # Waveform
            st.markdown("### 🌊 Waveform")
            waveform_fig, ax = plt.subplots(figsize=(12,3))
            librosa.display.waveshow(y, sr=sr, ax=ax, color="#1f77b4")
            st.pyplot(waveform_fig)

            # Mel Spectrogram
            st.markdown("### 🎨 Mel Spectrogram")
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_fig, ax = plt.subplots(figsize=(12,4))
            img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="viridis")
            mel_fig.colorbar(img, ax=ax, format="%+2.0f dB")
            st.pyplot(mel_fig)

            # Instrument intensity timeline (simulated as top instrument probability)
            st.markdown("### 📈 Instrument Intensity Timeline")
            timeline_fig, ax = plt.subplots(figsize=(12,3))
            top_inst = list(predictions.keys())[0]
            prob_vals = [predictions[top_inst]]*len(y)
            ax.plot(np.linspace(0, len(y)/sr, len(prob_vals)), prob_vals, color="#ff7f0e")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"{top_inst} Confidence")
            timeline_fig.tight_layout()
            st.pyplot(timeline_fig)

        # ==========================
        # EXPORT BUTTONS
        # ==========================
        st.subheader("📦 Export Reports")
        st.download_button("📄 Download JSON", json.dumps(predictions, indent=4), file_name="instrument_report.json", mime="application/json")
        pdf_bytes = generate_pdf(predictions, uploaded_file.name, waveform_fig, mel_fig, timeline_fig)
        st.download_button("📑 Download PDF", pdf_bytes, file_name="instrument_report.pdf", mime="application/pdf")

        os.remove(temp_path)

# ==========================
# ENTRY POINT
# ==========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_page()
else:
    main_app()
