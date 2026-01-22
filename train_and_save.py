import os
import json
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

# ==========================
# CONFIG
# ==========================
DATASET_PATH = r"C:\Users\manoj\Desktop\CNN-Based-Music-Instrument-Recognition-System\datasets"

SAMPLE_RATE = 22050
DURATION = 2.5
N_MELS = 64
MAX_FRAMES = 87
EPOCHS = 30
BATCH_SIZE = 16

# ==========================
# LOAD DATA
# ==========================
instruments = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

label_map = {inst: idx for idx, inst in enumerate(instruments)}
num_classes = len(instruments)

X, y = [], []

print("🔄 Loading dataset...")

for inst in instruments:
    folder = os.path.join(DATASET_PATH, inst)

    for file in os.listdir(folder):
        if not file.endswith(".wav"):
            continue

        path = os.path.join(folder, file)
        audio, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION)

        if len(audio) < int(DURATION * sr):
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

        mel_db = mel_db[..., np.newaxis]
        X.append(mel_db)

        label = np.zeros(num_classes)
        label[label_map[inst]] = 1
        y.append(label)

X = np.array(X)
y = np.array(y)

print("✅ Dataset loaded:", X.shape, y.shape)

# Save label map
with open("label_map.json", "w") as f:
    json.dump(label_map, f)

# ==========================
# SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# MODEL
# ==========================
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", padding="same",
           input_shape=(N_MELS, MAX_FRAMES, 1)),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(num_classes, activation="sigmoid")
])

model.compile(
    optimizer=Adam(0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================
# TRAIN
# ==========================
model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)

# ==========================
# SAVE (CRITICAL FIX)
# ==========================
# Save the trained model in HDF5 format
model.save("cnn_music_instruments.h5")
print("✅ Model saved as cnn_music_instruments.h5")

