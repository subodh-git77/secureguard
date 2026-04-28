# import torch
# import cv2
# import numpy as np
# import os
# from PIL import Image

# # Import the PyTorch loader from your local module
# from .model import load_model as load_pytorch_model 
# from .audio_utils import audio_to_spec

# # Import the Keras loader with an alias
# from tensorflow.keras.models import load_model as load_keras_model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- AUDIO MODEL SETUP (PyTorch) ---
# audio_model = load_pytorch_model() 
# # Using raw string for Windows paths to avoid escape character issues
# audio_model_path = os.path.join("saved_model", "deepfake_audio_model.pth")
# audio_model.load_state_dict(torch.load(audio_model_path, map_location=device))
# audio_model.to(device)
# audio_model.eval()

# # --- IMAGE MODEL SETUP (Keras/TensorFlow) ---
# IMG_SIZE = 224
# # Set threshold to 0.5 for standard probability. 
# # If your model is "flipped", we swap the labels in the logic below.
# THRESHOLD = 0.5 

# # Robust Path Discovery
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# img_model_path = os.path.join(BASE_DIR, "saved_model", "deepfake_cnn_model.keras")

# # Defensive Loading
# if not os.path.exists(img_model_path):
#     raise FileNotFoundError(f"CRITICAL: Model file not found at {img_model_path}")

# try:
#     image_model = load_keras_model(img_model_path)
# except Exception as e:
#     print("-" * 50)
#     print(f"FAILED TO LOAD KERAS MODEL: {e}")
#     print("TIP: If this is a PyTorch file renamed to .keras, this will fail.")
#     print("-" * 50)
#     raise e

# def predict_audio(path):
#     spec = audio_to_spec(path)
#     spec = spec.unsqueeze(0).to(device)

#     with torch.no_grad():
#         output = audio_model(spec)
#         probs = torch.softmax(output, dim=1)
#         pred = torch.argmax(output, 1).item()
#         confidence = probs[0][pred].item()

#     # FIX: If Audio is flipped (predicting Real when Fake), swap these strings
#     label = "REAL" if pred == 0 else "FAKE"
#     return label, confidence

# def predict_image(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         raise ValueError(f"Image not found at {image_path}")

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)

#     # Get probability from Keras model
#     prediction_score = image_model.predict(img)[0][0]

#     # --- LABEL CORRECTION LOGIC ---
#     # If your model is predicting Fake as Real, swap the "REAL" and "FAKE" strings below.
#     if prediction_score > THRESHOLD:
#         label = "FAKE"  # Changed from your previous script to try and fix the flip
#         confidence = prediction_score
#     else:
#         label = "REAL"
#         confidence = 1 - prediction_score
     
#     return label, float(confidence)


import torch
import cv2
import numpy as np
import tensorflow as tf
import os
from PIL import Image

# Import local modules
from .model import load_model as load_pytorch_model 
from .audio_utils import audio_to_spec

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- AUDIO MODEL SETUP (PyTorch) ---
audio_model = load_pytorch_model() 
audio_model_path = os.path.join("saved_model", "deepfake_audio_model.pth")
audio_model.load_state_dict(torch.load(audio_model_path, map_location=device))
audio_model.to(device)
audio_model.eval()

# --- IMAGE MODEL SETUP (TensorFlow SavedModel) ---
IMG_SIZE = 224
THRESHOLD = 0.15 # Using the threshold you mentioned earlier

# Robust Path Discovery - Pointing to the FOLDER now
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Update this path to match your extracted folder name
img_model_path = os.path.join(BASE_DIR, "saved_model", "saved_model", "saved_model_folder")

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_audio_features(audio_path):

    mel = audio_to_spec(audio_path)

    mel = mel.unsqueeze(0).to(device)   # [1,3,224,224]

    with torch.no_grad():

        x = audio_model.features(mel)

        # global pooling
        x = F.adaptive_avg_pool2d(x, 1)

        audio_features = torch.flatten(x, 1)

    return audio_features
# Defensive Loading for SavedModel Folder
if not os.path.exists(img_model_path):
    raise FileNotFoundError(f"CRITICAL: Model folder not found at {img_model_path}")

try:
    # Use the robust loader we built
    image_model = tf.keras.models.load_model(img_model_path, compile=False)
    print("✅ Image model loaded successfully for Audio Inference!")
except Exception as e:
    print(f"⚠️ Falling back to raw TF load: {e}")
    image_model = tf.saved_model.load(img_model_path)


def predict_audio(path):
    spec = audio_to_spec(path)
    spec = spec.unsqueeze(0).to(device)

    with torch.no_grad():
        output = audio_model(spec)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(output, 1).item()
        confidence = probs[0][pred].item()

    # Audio Label Logic
    label = "REAL" if pred == 0 else "FAKE"
    return label, confidence

def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # --- FIX: CALLING MODEL DIRECTLY ---
    # Convert to tensor and call directly to avoid AttributeError: '_UserObject' has no attribute 'predict'
    img_tensor = tf.convert_to_tensor(img)
    
    try:
        # High-level call
        prediction_score = image_model(img_tensor, training=False).numpy()[0][0]
    except:
        # Signature fallback
        infer = image_model.signatures["serving_default"]
        input_key = list(infer.structured_input_signature[1].keys())[0]
        output = infer(**{input_key: img_tensor})
        output_key = list(output.keys())[0]
        prediction_score = output[output_key].numpy()[0][0]

    # --- LABEL LOGIC ---
    # Based on our previous tests: score near 0 means FAKE, score near 1 means REAL
    if prediction_score > THRESHOLD:
        label = "REAL"
        confidence = prediction_score
    else:
        label = "FAKE"
        confidence = 1 - prediction_score
     
    return label, float(confidence)