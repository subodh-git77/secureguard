from src.video_utils import load_video_model, predict_video

import os

MODEL_PATH = os.path.join("saved_model", "best_deepfake_model.pth")
VIDEO_PATH = "id0_id4_0006.mp4"

model = load_video_model(MODEL_PATH)

label, confidence = predict_video(VIDEO_PATH, model)

print("Prediction:", label)
print("Confidence:", round(confidence,3))