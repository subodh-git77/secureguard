# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from torchvision import transforms
# from facenet_pytorch import MTCNN
# import timm
# import torch.nn as nn


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # load model
# def load_video_model(model_path):

#     model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)

#     model.load_state_dict(torch.load(model_path, map_location=device))

#     model = model.to(device)
#     model.eval()

#     return model


# # face detector
# mtcnn = MTCNN(keep_all=False, device=device)


# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],
#                          [0.229,0.224,0.225])
# ])


# def predict_video(video_path, model, frame_skip=3, threshold=0.85):

#     cap = cv2.VideoCapture(video_path)
#     predictions = []
#     frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_count % frame_skip == 0:
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(rgb)

#             # Detect face
#             face = mtcnn(image)

#             if face is not None:
#                 # Permute and normalize for PIL conversion
#                 face = face.permute(1, 2, 0).cpu().numpy()
#                 face = (face - face.min()) / (face.max() - face.min() + 1e-5)
#                 face_img = Image.fromarray((face * 255).astype("uint8"))

#                 # Apply transforms for the model
#                 img = transform(face_img).unsqueeze(0).to(device)

#                 with torch.no_grad():
#                     outputs = model(img)
#                     probs = torch.softmax(outputs, dim=1)

#                     # --- FIX START ---
#                     # If your model is flipping Real/Fake, swap these indices:
#                     # Index 0 is usually 'Fake' in many datasets, but if it's flipped for you:
#                     # Use probs[0][0] if class 0 is Fake, probs[0][1] if class 1 is Fake.
                    
#                     fake_prob = probs[0][0].item()  # Try changing [0] to [1] if results are still flipped
#                     predictions.append(fake_prob)
#                     # --- FIX END ---

#         frame_count += 1

#     cap.release()

#     if len(predictions) == 0:
#         return "No face detected", 0

#     # Calculate the average probability of the video being FAKE
#     avg_fake_score = np.mean(predictions)

#     # Decision Logic
#     if avg_fake_score > threshold:
#         label = "FAKE"
#         confidence = avg_fake_score
#     else:
#         label = "REAL"
#         confidence = 1 - avg_fake_score

#     return label, round(float(confidence), 3)


import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
import timm

import os

FACES_DIR = "static/faces"
os.makedirs(FACES_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ model path (UPDATE THIS)
MODEL_PATH = "saved_model/best_deepfake_model.pth"


import cv2
import torch
import numpy as np

import torch
import torch.nn.functional as F
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_video_features(video_path):

    print("start video feature extraction")

    cap = cv2.VideoCapture(video_path)

    success, frame = cap.read()

    if not success:
        raise ValueError("Cannot read video frame")

    # resize
    frame = cv2.resize(frame, (224,224))

    # convert BGR → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # normalize
    frame = frame / 255.0

    frame = torch.tensor(frame).permute(2,0,1).float()

    frame = frame.unsqueeze(0).to(device)   # [1,3,224,224]

    with torch.no_grad():

        x = video_model.forward_features(frame)

        x = F.adaptive_avg_pool2d(x,1)

        video_features = torch.flatten(x,1)

    print("Video feature extracted!")

    return video_features
# load model
def load_video_model(model_path):

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model.eval()

    return model


# ✅ LOAD MODEL ONCE (IMPORTANT FOR FLASK)
video_model = load_video_model(MODEL_PATH)


# face detector
mtcnn = MTCNN(keep_all=False, device=device)


# ✅ KEEP YOUR NORMALIZATION (CORRECT)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])


# ✅ YOUR FUNCTION (UNCHANGED LOGIC)
def predict_video(video_path, model=video_model, frame_skip=3, threshold=0.85):

    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0

    # 🔥 NEW: store face image names
    face_paths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            face = mtcnn(image)

            if face is not None:
                face = face.permute(1, 2, 0).cpu().numpy()
                face = (face - face.min()) / (face.max() - face.min() + 1e-5)
                face_img = Image.fromarray((face * 255).astype("uint8"))

                # 🔥 NEW: SAVE FACE
                face_filename = f"face_{frame_count}.jpg"
                face_path = os.path.join(FACES_DIR, face_filename)
                face_img.save(face_path)

                face_paths.append(face_filename)

                img = transform(face_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(img)
                    probs = torch.softmax(outputs, dim=1)

                    fake_prob = probs[0][0].item()
                    predictions.append(fake_prob)

        frame_count += 1

    cap.release()

    if len(predictions) == 0:
        return "No face detected", 0, 0, []

    avg_fake_score = np.mean(predictions)

    if avg_fake_score > threshold:
        label = "FAKE"
        confidence = avg_fake_score
    else:
        label = "REAL"
        confidence = 1 - avg_fake_score

    # 🔥 UPDATED RETURN
    return label, round(float(confidence), 3), len(predictions), face_paths