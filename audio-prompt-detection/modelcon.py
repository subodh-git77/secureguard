import os
# Force Keras to use TensorFlow as the backend
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf

# Define your file paths
input_path = r"audio-prompt-detection\saved_model\best_deepfake_model.pth"  # The file causing the error
output_path = r"audio-prompt-detection\saved_model\deepfake_image_model.h5"   # The new file we will create

print(f"Loading {input_path} using Keras {keras.__version__}...")

try:
    # Load the Keras 3 model
    # We use safe_mode=False in case there are custom layers
    model = keras.models.load_model(input_path, safe_mode=False)
    
    # Save it in the older H5 format that TF 2.13 understands
    model.save(output_path)
    
    print("-" * 30)
    print(f"SUCCESS! Model converted to: {output_path}")
    print("You can now load 'fixed_model.h5' in your main project.")
    print("-" * 30)

except Exception as e:
    print(f"Conversion failed: {e}")