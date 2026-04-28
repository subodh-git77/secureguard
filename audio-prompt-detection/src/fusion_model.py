
import torch
import torch.nn as nn
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definition ---
# Note: Keep the architecture exactly as you provided
fusion_model = nn.Sequential(
    nn.Linear(1793, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

# Load weights
model_path = "saved_model/fusion_model.pth"
if os.path.exists(model_path):
    fusion_model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
fusion_model.to(device)
fusion_model.eval()

def fusion_prediction(audio_features, video_features, image_features):
    # CORRECTED: Use the variables passed as arguments, NOT the imported functions
    # Also, ensure they have a batch dimension [1, features] if they don't already
    
    if audio_features.dim() == 1:
        audio_features = audio_features.unsqueeze(0)
    if video_features.dim() == 1:
        video_features = video_features.unsqueeze(0)
    if image_features.dim() == 1:
        image_features = image_features.unsqueeze(0)

    # Move tensors to the correct device
    audio_features = audio_features.to(device)
    video_features = video_features.to(device)
    image_features = image_features.to(device)

    # Concatenate features along the feature dimension (dim=1)
    combined_features = torch.cat(
        (audio_features, video_features, image_features),
        dim=1
    )

    with torch.no_grad():
        prediction = fusion_model(combined_features)

    prob = prediction.item()

    # Logic for result and confidence
    result = "FAKE" if prob > 0.5 else "REAL"
    
    # If prob is 0.9, FAKE confidence is 0.9
    # If prob is 0.1, REAL confidence is 0.9 (1 - 0.1)
    confidence = prob if result == "FAKE" else 1.0 - prob

    return result, confidence