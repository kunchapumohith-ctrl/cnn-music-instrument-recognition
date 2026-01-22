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
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from fpdf import FPDF
import tempfile

# ==========================
# LOGIN CREDENTIALS
# ==========================
CREDENTIALS = {
    "admin": "admin123"
}

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

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
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH, compile=False)

model = load_cnn_model()

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}

# ==========================
# LOGIN PAGE
# ==========================
def login_page():
    st.markdown("<h1 style='color:#4CAF50'>InstruNet AI</h1>", unsafe_allow_html=True)
    st.markdown("### CNN Based Music Instrument Recognition System")
    st.info("Please login to continue")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if username in CREDENTIALS and CREDENTIALS[username] == password:
                st.session_state.authenticated = True
                st.success("Login successful. Reloading app...")
                st.rerun()
            else:
                st.error("Incorrect username or password")

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
        mel_db = np.pad(mel_db, ((0, 0), (0, MAX_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_FRAMES]

    mel_db = mel_db[np.newaxis, ..., np.newaxis]
    probs = model.predict(mel_db, verbose=0)[0]

    top_idx = np.argsort(probs)[::-1][:TOP_K]
    return {inv_label_map[i]: float(probs[i]) for i in top_idx}, audio, sr, mel_db

# ==========================
# SAVE VISUALIZATIONS
# ==========================
def save_plot(fig):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

# ==========================
# PDF GENERATION
# ==========================
def generate_pdf(predictions, image_paths):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "InstruNet AI - Instrument Recognition Report", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Detected Instruments:", ln=True)

    pdf.set_font("Arial", size=11)
    for inst, conf in predictions.items():
        pdf.cell(0, 7, f"{inst}: {conf*100:.2f}%", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Visual Analysis:", ln=True)

    for img in image_paths:
        pdf.add_page()
        pdf.image(img, x=10, y=20, w=180)

    return pdf.output(dest="S").encode("latin1")

# ==========================
# MAIN APP
# ==========================
def main_app():
    st.markdown("<h1 style='color:#4CAF50'>InstruNet AI</h1>", unsafe_allow_html=True)
    st.markdown("### CNN Based Music Instrument Recognition System")
    st.success("Welcome! Upload an audio file to begin analysis.")

    uploaded = st.file_uploader("Upload WAV audio file", type=["wav"])

    if uploaded:
        with open("temp.wav", "wb") as f:
            f.write(uploaded.getbuffer())

        predictions, audio, sr, mel_db = predict_instruments("temp.wav")

        st.subheader("Predicted Instruments")
        for inst, conf in predictions.items():
            st.write(f"{inst}: {conf*100:.2f}%")
            st.progress(int(conf * 100))

        if st.button("Show Visualizations"):
            figs = []

            fig1, ax1 = plt.subplots()
            librosa.display.waveshow(audio, sr=sr, ax=ax1)
            ax1.set_title("Waveform")
            st.pyplot(fig1)
            figs.append(save_plot(fig1))

            fig2, ax2 = plt.subplots()
            img = librosa.display.specshow(mel_db[0, :, :, 0], sr=sr, ax=ax2)
            fig2.colorbar(img, ax=ax2)
            ax2.set_title("Mel Spectrogram")
            st.pyplot(fig2)
            figs.append(save_plot(fig2))

            fig3, ax3 = plt.subplots()
            ax3.bar(predictions.keys(), predictions.values())
            ax3.set_title("Instrument Intensity Timeline")
            st.pyplot(fig3)
            figs.append(save_plot(fig3))

            st.download_button(
                "Download JSON Report",
                json.dumps(predictions, indent=4),
                file_name="instrument_report.json",
                mime="application/json"
            )

            pdf_bytes = generate_pdf(predictions, figs)
            st.download_button(
                "Download PDF Report",
                pdf_bytes,
                file_name="InstruNet_Report.pdf",
                mime="application/pdf"
            )

# ==========================
# ENTRY POINT
# ==========================
if not st.session_state.authenticated:
    login_page()
else:
    main_app()
