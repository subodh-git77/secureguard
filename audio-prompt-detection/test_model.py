from keras.models import load_model

model = load_model("saved_model/deepfake_cnn_model.keras", compile=False)

print("Model loaded successfully")