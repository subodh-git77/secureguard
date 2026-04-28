from src.inference import predict_audio

audio_path = "temp_audio.wav"

label, confidence = predict_audio(audio_path)

print("Prediction:", label)
print("Confidence:", confidence)