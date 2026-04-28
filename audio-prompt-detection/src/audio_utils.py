# import librosa
# from src.config import SAMPLE_RATE, DURATION

# def load_audio(path):
#     audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
#     audio = librosa.util.fix_length(audio, size=SAMPLE_RATE * DURATION)
#     return audio

import librosa
import numpy as np
import cv2
import torch
import subprocess
import os

def audio_to_spec(path):

    audio, sr = librosa.load(path, sr=16000)

    max_len = 16000 * 2

    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)))
    else:
        audio = audio[:max_len]

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )

    spec = librosa.power_to_db(spec, ref=np.max)

    spec = cv2.resize(spec, (224,224))

    spec = np.stack((spec,)*3, axis=0)

    spec = torch.tensor(spec).float()

    return spec





# def extract_audio(video_path, output_path=None):

#     # auto output name
#     if output_path is None:
#         base = os.path.splitext(video_path)[0]
#         output_path = base + "_audio.wav"

#     # ffmpeg command
#     command = [
#         "ffmpeg",
#         "-i", video_path,
#         "-vn",              # no video
#         "-acodec", "pcm_s16le",
#         "-ar", "16000",     # sample rate (important for ML)
#         "-ac", "1",         # mono audio
#         output_path,
#         "-y"
#     ]

#     try:
#         subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     except Exception as e:
#         print(f"❌ FFmpeg error: {e}")
#         return None

#     # check file created
#     if not os.path.exists(output_path):
#         print("❌ Audio extraction failed")
#         return None

#     return output_path
import subprocess
import os

def extract_audio_safe(video_path):

    audio_path = video_path.replace(".mp4","_audio.wav")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec","pcm_s16le",
        "-ar","16000",
        "-ac","1",
        audio_path,
        "-y"
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if os.path.exists(audio_path):
            return audio_path
        else:
            return None

    except:
        return None
