# # import os
# # from flask import Flask, render_template, request
# # from werkzeug.utils import secure_filename

# # # 🔥 import from your working inference
# # from src.inference import predict_audio, predict_image
# # from src.video_utils import predict_video
# # from src.audio_utils import audio_to_spec,extract_audio  # only for combined

# # app = Flask(__name__)

# # UPLOAD_FOLDER = "uploads"
# # os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# # @app.route("/")
# # def home():
# #     return render_template("index.html")


# # # ---------------- SINGLE INPUT ----------------
# # @app.route("/predict_single", methods=["POST"])
# # def predict_single():

# #     file = request.files["file"]

# #     if file.filename == "":
# #         return render_template("index.html", result="No file selected")

# #     filename = secure_filename(file.filename)
# #     path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
# #     file.save(path)

# #     ext = filename.split(".")[-1].lower()

# #     try:
# #         # IMAGE
# #         if ext in ["jpg", "jpeg", "png"]:
# #             label, conf = predict_image(path)

# #         # AUDIO
# #         elif ext in ["wav", "mp3"]:
# #             label, conf = predict_audio(path)

# #         # VIDEO
# #         elif ext == "mp4":
# #             label, conf,frame_count,faces = predict_video(path)

# #         else:
# #             return render_template("index.html",
# #                                    result="Unsupported file")

# #         return render_template("result.html",
# #                                result=label,
# #                                confidence=round(conf * 100, 2),
# #                                frames=frame_count,
# #                                faces=faces)

# #     except Exception as e:
# #         return render_template("index.html",
# #                                result=f"Error: {str(e)}")


# # # ---------------- COMBINED ----------------
# # @app.route("/predict_combined", methods=["POST"])
# # def predict_combined():

# #     file = request.files["file"]

# #     if file.filename == "":
# #         return render_template("index.html", result="No file selected")

# #     filename = secure_filename(file.filename)
# #     path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
# #     file.save(path)

# #     if not filename.endswith(".mp4"):
# #         return render_template("index.html",
# #                                result="Only MP4 supported")

# #     try:
# #     # extract audio
# #         audio_path = extract_audio(path)

# #         # predictions
# #         video_label, video_conf,frame_count,faces = predict_video(path)
# #         audio_label, audio_conf = predict_audio(audio_path)

# #         # 🔥 RULE-BASED FUSION
# #         if video_label == "FAKE" or audio_label == "FAKE":
# #             label = "FAKE"
# #             confidence = max(video_conf, audio_conf)
# #         else:
# #             label = "REAL"
# #             confidence = min(video_conf, audio_conf)

# #         return render_template("result.html",
# #                             result=label,
# #                             confidence=round(confidence * 100, 2),
# #                             frames=frame_count,  
# #                             faces=faces)

# #     except Exception as e:
# #         return render_template("index.html",
# #                                result=f"Error: {str(e)}")


# # # ---------------- MAIN ----------------
# # if __name__ == "__main__":
# #     app.run(debug=True)

# import os
# from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename

# # 🔥 import from your working inference
# from src.inference import predict_audio, predict_image
# from src.video_utils import predict_video
# from src.audio_utils import extract_audio_safe
# from src.video_utils import  get_video_features
# from src.inference import get_audio_features
# from src.fusion_model import fusion_model,fusion_prediction
# from src.agent import DeepfakeDetectionAgent


# app = Flask(__name__)

# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# import torch
# import tensorflow as tf

# device = "cuda" if torch.cuda.is_available() else "cpu"
# agent=DeepfakeDetectionAgent(device)

# # audio + video models
# # audio_model = predict_audio()
# # video_model = predict_video()

# # image model
# image_model = tf.saved_model.load(
#     r"E:\new_hack_iitk\audio-prompt-detection\saved_model\saved_model\saved_model_folder"
# )
# # fusion model


# # ---------------- HOME (Landing Page) ----------------
# @app.route("/")
# def home():
#     return render_template("index.html")


# # ---------------- UPLOAD PAGE ----------------
# @app.route("/upload")
# def upload_page():
#     return render_template("upload.html")


# # ---------------- SINGLE INPUT ----------------
# @app.route("/predict_single", methods=["POST"])
# def predict_single():

#     file = request.files.get("file")

#     if not file or file.filename == "":
#         return render_template("index.html", result="No file selected")

#     filename = secure_filename(file.filename)
#     path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#     file.save(path)

#     ext = filename.split(".")[-1].lower()

#     # ✅ SAFE DEFAULTS (IMPORTANT)
#     frame_count = None
#     faces = None

#     try:
#         # IMAGE
#         if ext in ["jpg", "jpeg", "png"]:
#             label, conf = predict_image(path)

#         # AUDIO
#         elif ext in ["wav", "mp3"]:
#             label, conf = predict_audio(path)

#         # VIDEO
#         elif ext == "mp4":
#             label, conf, frame_count, faces = predict_video(path)

#         else:
#             return render_template("index.html",
#                                    result="Unsupported file")

#         return render_template("result.html",
#                                result=label,
#                                confidence=round(conf * 100, 2),
#                                frames=frame_count,
#                                faces=faces)

#     except Exception as e:
#         return render_template("index.html",
#                                result=f"Error: {str(e)}")


# # ---------------- COMBINED ----------------
# # @app.route("/predict_combined", methods=["POST"])
# # def predict_combined():

# #     file = request.files.get("file")

# #     if not file or file.filename == "":
# #         return render_template("index.html", result="No file selected")

# #     filename = secure_filename(file.filename)
# #     path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
# #     file.save(path)

# #     if not filename.endswith(".mp4"):
# #         return render_template("index.html",
# #                                result="Only MP4 supported")

# #     try:
# #         # 🔥 extract audio
# #         audio_path = extract_audio(path)

# #         # predictions
# #         video_label, video_conf, frame_count, faces = predict_video(path)
# #         audio_label, audio_conf = predict_audio(audio_path)

# #         # 🔥 RULE-BASED FUSION (your logic kept)
# #         if video_label == "FAKE" or audio_label == "FAKE":
# #             label = "FAKE"
# #             confidence = max(video_conf, audio_conf)
# #         else:
# #             label = "REAL"
# #             confidence = min(video_conf, audio_conf)

# #         return render_template("result.html",
# #                                result=label,
# #                                confidence=round(confidence * 100, 2),
# #                                frames=frame_count,
# #                                faces=faces)

# #     except Exception as e:
# #         return render_template("index.html",
# #                                result=f"Error: {str(e)}")
# # @app.route("/predict_combined", methods=["POST"])
# # def predict_combined():

# #     file = request.files.get("file")

# #     if not file or file.filename == "":
# #         return render_template("index.html", result="No file selected")

# #     filename = secure_filename(file.filename)
# #     path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
# #     file.save(path)

# #     if not filename.endswith(".mp4"):
# #         return render_template("index.html", result="Only MP4 supported")

# #     try:

# #         # 1️⃣ extract audio from video
# #         audio_path = extract_audio_safe(path)
# #         print("Audio extracted:",audio_path)

# #         # 2️⃣ get video features
# #         print("start video feature extraction")
# #         # VIDEO FEATURES
# #         video_features = get_video_features(path)

# #         # AUDIO FEATURES
# #         audio_path = extract_audio_safe(path)

# #         if audio_path is None:
# #             audio_features = torch.ones((1,512)) * 0.01
# #         else:
# #             audio_features = get_audio_features(audio_path)

# #         # IMAGE FEATURE
# #         image_features = torch.tensor([[0.5]]).float().to(device)

# #         print(audio_features.shape)
# #         print(video_features.shape)
# #         print(image_features.shape)
# #         print("Audio mean:", audio_features.mean().item())
# #         print("Video mean:", video_features.mean().item())
# #         print("Image value:", image_features.item())

# #         # 5️⃣ combine features
# #         combined_features = torch.cat(
# #             (audio_features, video_features, image_features),
# #             dim=1
# #         )
# #         print("Combined shape:", combined_features.shape)
# #         # 6️⃣ fusion prediction
# #         with torch.no_grad():
# #             prediction = fusion_model(combined_features)

# #         print("Fusion raw output:", prediction.item())

# #         print("---------------------------")

# #         prob = prediction.item()

# #         if prob > 0.65:
# #             label = "FAKE"
# #         else:
# #             label = "REAL"

# #         confidence = prob if label == "FAKE" else 1 - prob

# #         # still show video statistics
# #         _, _, frame_count, faces = predict_video(path)

# #         return render_template("result.html",
# #                                result=label,
# #                                confidence=round(confidence * 100, 2),
# #                                frames=frame_count,
# #                                faces=faces)

# #     except Exception as e:
# #         print("ERROR OCCURRED:",e)
# #         return str(e)

# @app.route("/predict_combined", methods=["POST"])
# def predict_combined():

#     file = request.files.get("file")

#     if not file or file.filename == "":
#         return render_template("index.html", result="No file selected")

#     filename = secure_filename(file.filename)
#     path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#     file.save(path)

#     if not filename.endswith(".mp4"):
#         return render_template("index.html", result="Only MP4 supported")

#     try:

#         output = agent.run_pipeline(path)

#         return render_template(
#             "result.html",
#             result=output["result"],
#             confidence=round(output["confidence"] * 100, 2),
#             frames=output["frames"],
#             faces=output["faces"]
#         )

#     except Exception as e:

#         return render_template(
#             "index.html",
#             result=f"Error: {str(e)}"
#         )
# # ---------------- MAIN ----------------
# if __name__ == "__main__":
#     app.run(debug=True)

import os
import torch
import tensorflow as tf

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.agent import DeepfakeAgent

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Agent
agent = DeepfakeAgent(device)

# Load image model once
image_model = tf.saved_model.load(
    r"C:\new_hack_iitk\audio-prompt-detection\saved_model\saved_model\saved_model_folder"
)

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- UPLOAD PAGE ----------------
@app.route("/upload")
def upload_page():
    return render_template("upload.html")

# ---------------- AGENT PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("file")

    if not file or file.filename == "":
        return render_template("index.html", result="No file selected")

    filename = secure_filename(file.filename)

    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(path)

    try:

        # Agent handles everything
        output = agent.run(path)

        return render_template(
            "result.html",
            result=output["result"],
            confidence=round(output["confidence"] * 100, 2),
            frames=output["frames"],
            faces=output["faces"]
        )

    except Exception as e:
            print("Error occured:",e)
            return str(e)
        

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True)