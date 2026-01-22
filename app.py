import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import json
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
from tensorflow.keras.models import load_model

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="InstruNet AI", layout="wide")

# ==========================
# MODEL & LABEL MAP
# ==========================
MODEL_PATH = "cnn_music_instruments.h5"
LABEL_MAP_PATH = "label_map.json"
SAMPLE_RATE = 22050
DURATION = 2.5
N_MELS = 64
MAX_FRAMES = 87
TOP_K = 3

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH, compile=False)

try:
    model = load_cnn_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)
inv_label_map = {v:k for k,v in label_map.items()}

# ==========================
# SESSION STATE
# ==========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "visuals" not in st.session_state:
    st.session_state.visuals = {}

# ==========================
# LOGIN PAGE
# ==========================
USERNAME = "user"
PASSWORD = "instrunet123"

def login_page():
    st.markdown("<h1 style='color:#4B0082;text-align:center;'>🎵 InstruNet AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#6A5ACD;text-align:center;'>CNN-Based Music Instrument Recognition System</h3><br>", unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == USERNAME and password == PASSWORD:
                st.session_state.authenticated = True
                st.session_state.user = username
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Incorrect username or password")

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_instruments(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(audio) < DURATION*sr:
        audio = np.pad(audio, (0,int(DURATION*sr)-len(audio)))
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel)
    if mel_db.shape[1] < MAX_FRAMES:
        mel_db = np.pad(mel_db, ((0,0),(0,MAX_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :MAX_FRAMES]
    mel_db = mel_db[np.newaxis,...,np.newaxis]
    probs = model.predict(mel_db, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:TOP_K]
    results = {inv_label_map[i]: float(probs[i]) for i in top_indices}
    return results

# ==========================
# PDF REPORT FUNCTION
# ==========================
def generate_pdf_report(predictions, audio_name, visuals):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "InstruNet AI Instrument Recognition Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(0,8,f"Audio File: {audio_name}", ln=True)
    pdf.cell(0,8,f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    # Detected Instruments
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0,8,"Detected Instruments:", ln=True)
    pdf.set_font("Arial", size=12)
    for inst, conf in predictions.items():
        pdf.cell(80, 8, inst, border=1)
        pdf.cell(40,8,f"{conf*100:.2f}%", border=1, ln=True)

    # Add visualizations
    for vis_name, fig in visuals.items():
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0,8, vis_name, ln=True)
        image_path = f"temp_{vis_name}.png"
        fig.savefig(image_path)
        pdf.image(image_path, x=10, y=20, w=pdf.w - 20)
        os.remove(image_path)

    return pdf.output(dest="S").encode("latin1")

# ==========================
# MAIN APP
# ==========================
def main_app():
    st.markdown("<h1 style='color:#4B0082;'>🎵 InstruNet AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#6A5ACD;'>CNN-Based Music Instrument Recognition System</h3><br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])

    if uploaded_file is not None:
        temp_path = "temp.wav"
        with open(temp_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(temp_path)

        try:
            predictions = predict_instruments(temp_path)
            st.session_state.predictions = predictions
            st.session_state.audio_data = uploaded_file.name

            st.subheader("🎯 Top Predicted Instruments")
            for inst, conf in predictions.items():
                st.write(f"**{inst}** : {conf*100:.2f}%")
                st.progress(int(conf*100))

            # Visualization button
            if st.button("📊 Visualizations"):
                st.subheader("🌊 Waveform")
                y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
                fig_wav, ax = plt.subplots(figsize=(10,2))
                librosa.display.waveshow(y, sr=sr, ax=ax)
                st.pyplot(fig_wav)
                st.session_state.visuals['Waveform'] = fig_wav

                st.subheader("🎨 Mel Spectrogram")
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                fig_mel, ax = plt.subplots(figsize=(10,3))
                librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
                fig_mel.colorbar(ax=ax)
                st.pyplot(fig_mel)
                st.session_state.visuals['Mel Spectrogram'] = fig_mel

                st.subheader("📈 Instrument Intensity Timeline")
                fig_intensity, ax = plt.subplots(figsize=(10,3))
                instruments = list(predictions.keys())
                values = list(predictions.values())
                ax.bar(instruments, values, color='#6A5ACD')
                ax.set_ylabel('Confidence')
                st.pyplot(fig_intensity)
                st.session_state.visuals['Intensity Timeline'] = fig_intensity

            # Export buttons
            st.subheader("📤 Export Results")
            st.download_button("📥 Download JSON", json.dumps(predictions, indent=4), file_name="instrument_report.json", mime="application/json")

            pdf_bytes = generate_pdf_report(predictions, uploaded_file.name, st.session_state.visuals)
            st.download_button("📄 Download PDF Report", pdf_bytes, file_name="instrument_report.pdf", mime="application/pdf")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

        os.remove(temp_path)

# ==========================
# ENTRY POINT
# ==========================

if not st.session_state.authenticated:
    login_page()
else:
    main_app()
