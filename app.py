import streamlit as st
import numpy as np
import librosa
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from fpdf import FPDF
import zipfile

# ==========================
# CONFIG
# ==========================
MODEL_DIR = "cnn_music_instruments"    # folder after unzip
MODEL_ZIP = "cnn_music_instruments.zip"
LABEL_MAP_PATH = "label_map.json"

SAMPLE_RATE = 22050
DURATION = 2.5
N_MELS = 64
MAX_FRAMES = 87
TOP_K = 3   # show top 3 instruments

# ==========================
# UNZIP MODEL IF NEEDED
# ==========================
if not os.path.exists(MODEL_DIR):
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall(MODEL_DIR)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_DIR)

model = load_cnn_model()

# ==========================
# LOAD LABEL MAP
# ==========================
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
    # label_map format: { "accordion": 0, "bass": 1, ... }

# invert → index → instrument
inv_label_map = {v: k for k, v in label_map.items()}

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_instruments(file_path, top_k=TOP_K):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    if len(audio) < DURATION * sr:
        audio = np.pad(audio, (0, int(DURATION * sr) - len(audio)))

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel)

    if mel_db.shape[1] < MAX_FRAMES:
        mel_db = np.pad(
            mel_db, ((0, 0), (0, MAX_FRAMES - mel_db.shape[1]))
        )
    else:
        mel_db = mel_db[:, :MAX_FRAMES]

    mel_db = mel_db[np.newaxis, ..., np.newaxis]

    probs = model.predict(mel_db, verbose=0)[0]

    # 🔥 TOP-K predictions
    top_indices = np.argsort(probs)[::-1][:top_k]

    results = {
        inv_label_map[i]: float(probs[i])
        for i in top_indices
    }

    return results

# ==========================
# PDF REPORT FUNCTION
# ==========================
def generate_pdf_report(predictions, audio_name):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "CNN-Based Music Instrument Recognition System", ln=True, align="C")

    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Audio File: {audio_name}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 8, "Top Predicted Instruments:", ln=True)

    pdf.ln(3)
    for inst, conf in predictions.items():
        pdf.cell(80, 8, inst, border=1)
        pdf.cell(40, 8, f"{conf*100:.2f}%", border=1, ln=True)

    return pdf.output(dest="S").encode("latin1")

# ==========================
# STREAMLIT UI
# ==========================
st.set_page_config(page_title="Music Instrument Recognition", layout="wide")
st.title("🎵 CNN-Based Music Instrument Recognition System")

uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])

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
            st.progress(int(conf * 100))

        # JSON download
        st.download_button(
            "📄 Download JSON Report",
            json.dumps(predictions, indent=4),
            file_name="instrument_report.json",
            mime="application/json"
        )

        # PDF download
        pdf_bytes = generate_pdf_report(predictions, uploaded_file.name)
        st.download_button(
            "📄 Download PDF Report",
            pdf_bytes,
            file_name="instrument_report.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("No instruments detected.")

    os.remove(temp_path)
