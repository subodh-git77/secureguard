import torch

from video_utils import get_video_features
from inference import get_audio_features
from audio_utils import extract_audio_safe
from fusion_model import fusion_prediction

device = "cuda" if torch.cuda.is_available() else "cpu"

video_path = "test.mp4"   # put a sample video here

print("STEP 1: Extract audio")
audio_path = extract_audio_safe(video_path)
print("Audio path:", audio_path)

print("\nSTEP 2: Video features")
video_features = get_video_features(video_path)
print("Type:", type(video_features))
print("Shape:", getattr(video_features, "shape", "NO SHAPE"))

print("\nSTEP 3: Audio features")
if audio_path is not None:
    audio_features = get_audio_features(audio_path)
else:
    audio_features = torch.zeros((1,512))

print("Type:", type(audio_features))
print("Shape:", getattr(audio_features, "shape", "NO SHAPE"))

print("\nSTEP 4: Image feature")
image_features = torch.tensor([[0.5]])

print("Shape:", image_features.shape)

print("\nSTEP 5: Move to device")

if isinstance(video_features, torch.Tensor):
    video_features = video_features.to(device)
else:
    print("ERROR: video_features is not tensor")

if isinstance(audio_features, torch.Tensor):
    audio_features = audio_features.to(device)
else:
    print("ERROR: audio_features is not tensor")

image_features = image_features.to(device)

print("\nSTEP 6: Fusion test")

try:
    result, confidence = fusion_prediction(
        audio_features,
        video_features,
        image_features
    )
    print("Fusion works!")
    print("Result:", result)
    print("Confidence:", confidence)

except Exception as e:
    print("Fusion error:", e)