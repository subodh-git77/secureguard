import cv2
import numpy as np
import tensorflow as tf
import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Ensure this path contains the .pb file
MODEL_DIR = r"C:\new_hack_iitk\audio-prompt-detection\saved_model\saved_model\saved_model_folder"


try:
    # Loading as a generic model to be safe across TF versions
    model = tf.keras.models.load_model(MODEL_DIR, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Loading via Keras failed, attempting raw TF load: {e}")
    model = tf.saved_model.load(MODEL_DIR)
    print("✅ Model loaded as Raw SavedModel.")

import cv2
import numpy as np
import torch

def get_image_features(image_path):

    img = cv2.imread(image_path)

    img = cv2.resize(img,(224,224))

    img = img / 255.0

    img = np.expand_dims(img, axis=0).astype(np.float32)

    preds = model(img)

    score = preds['output_0'].numpy()

    image_feature = torch.tensor(score).float()

    return image_feature



def predict_image(img_path):
    # Use raw string or fix pathing for Windows
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Error: Could not read image at {img_path}")
        return "Error", 0.0
    
    # Preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Convert image to a tensor for the computation graph
    img_tensor = tf.convert_to_tensor(img)

    # --- UPDATED INFERENCE LOGIC ---
    try:
        # Calling the model directly handles the '_UserObject' error
        # training=False is important for BatchNormalization layers
        predictions = model(img_tensor, training=False)
        pred = float(predictions.numpy()[0][0])
    except Exception as e:
        # Fallback for models with specific 'serving_default' signatures
        infer = model.signatures["serving_default"]
        # Automatically find the input layer name (usually 'input_1')
        input_name = list(infer.structured_input_signature[1].keys())[0]
        predictions = infer(**{input_name: img_tensor})
        output_name = list(predictions.keys())[0]
        pred = float(predictions[output_name].numpy()[0][0])
    
    # Label Logic
    threshold = 0.15
    label = "Fake" if pred > threshold else "Real"
    
    # Confidence is the distance from the non-predicted class
    confidence = pred if pred > threshold else (1 - pred)
    
    print(f"Prediction: {label} (Score: {pred:.4f})")
    return label, confidence