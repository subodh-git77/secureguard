

import torch
import os

from src.inference import predict_image, predict_audio, get_audio_features
from src.video_utils import predict_video, get_video_features
from src.audio_utils import extract_audio_safe
from src.fusion_model import fusion_prediction
from src.image_utils import get_image_features

class DeepfakeAgent:

    def __init__(self, device):
        self.device = device

    def detect_input_type(self, path):
        """Identifies if the input is an image, audio, or video based on extension."""
        ext = path.split(".")[-1].lower()
        if ext in ["jpg", "jpeg", "png"]:
            return "image"
        if ext in ["wav", "mp3"]:
            return "audio"
        if ext in ["mp4", "avi", "mov"]:
            return "video"
        return "unknown"

    def analyze_video(self, video_path):
        """Extracts audio from video and checks if it exists."""
        audio_path = extract_audio_safe(video_path)
        if audio_path is None:
            return {
                "has_audio": False,
                "audio_path": None
            }
        return {
            "has_audio": True,
            "audio_path": audio_path
        }

    def run(self, file_path):
        """Main entry point for detection logic."""
        input_type = self.detect_input_type(file_path)
        print(f"--- Agent detected input: {input_type} ---")

        # ---------------- IMAGE PIPELINE ----------------
        if input_type == "image":
            label, conf = predict_image(file_path)
            return {
                "result": label,
                "confidence": conf,
                "frames": None,
                "faces": None
            }

        # ---------------- AUDIO PIPELINE ----------------
        if input_type == "audio":
            label, conf = predict_audio(file_path)
            return {
                "result": label,
                "confidence": conf,
                "frames": None,
                "faces": None
            }

        # ---------------- VIDEO PIPELINE ----------------
        if input_type == "video":
            analysis = self.analyze_video(file_path)

            # Case A: Video without audio (Visual-only)
            if not analysis["has_audio"]:
                print("Agent decision: Video-only pipeline")
                label, conf, frames, faces = predict_video(file_path)
                return {
                    "result": label,
                    "confidence": conf,
                    "frames": frames,
                    "faces": faces
                }

            # Case B: Video with audio (Multimodal Fusion)
            print("Agent decision: Multimodal pipeline")
            audio_path = analysis["audio_path"]
            
            # 1. Feature Extraction
            print("Extracting features...")
            v_feat = get_video_features(file_path)
            a_feat = get_audio_features(audio_path)
            
            # Using a neutral image feature for the fusion model as per your original logic
            i_feat = torch.tensor([[0.5]]).float()

            # 2. Validation: Ensure we have Tensors before moving to device
            if not isinstance(v_feat, torch.Tensor) or not isinstance(a_feat, torch.Tensor):
                raise ValueError("Feature extraction failed to return a torch.Tensor")

            # 3. Move Tensors to Device
            # We use the local variables (v_feat, a_feat) NOT the function names
            video_features = v_feat.to(self.device)
            audio_features = a_feat.to(self.device)
            image_features = i_feat.to(self.device)

            print(f"Features moved to {self.device}")

            # 4. Multimodal Fusion Prediction
            print("Running fusion model...")
            result, confidence = fusion_prediction(
                audio_features,
                video_features,
                image_features
            )

            # 5. Extract additional visual stats for the UI/Return object
            _, _, frames, faces = predict_video(file_path)

            return {
                "result": result,
                "confidence": confidence,
                "frames": frames,
                "faces": faces
            }

        # ---------------- UNSUPPORTED ----------------
        return {
            "result": "Unsupported file format",
            "confidence": 0,
            "frames": None,
            "faces": None
        }