from tensorflow.keras.models import load_model

# load your existing trained model
model = load_model("cnn_music_instruments.keras", compile=False)

# save a deploy-safe version
model.save(
    "cnn_music_instruments_deploy.h5",
    include_optimizer=False
)

print("✅ Model saved as cnn_music_instruments_deploy.h5")
