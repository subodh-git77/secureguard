# 🔐 Multimodal Prompt Injection & Adversarial Media Detection

A real-time multimodal AI security system to detect **prompt injection**, **deepfake audio**, and **adversarial images**, and prevent unauthorized actions in AI-powered systems.

---

## 🚀 Overview

AI systems that accept audio and image inputs are vulnerable to:
- Spoken prompt injection attacks
- Deepfake or synthetic voice impersonation
- Adversarial images with hidden instructions

This project introduces a **multimodal defense framework** that independently analyzes **audio and image inputs**, detects malicious patterns, and enforces **risk-aware security actions** — **ALLOW**, **WARNING**, or **BLOCK** — before inputs reach downstream AI agents.

---

## 🧠 System Architecture


Each modality is processed **independently**, ensuring robustness, modularity, and clear evaluation.

---

## 🎧 Audio Detection

- Supported formats: `.wav`, `.mp3`, `.m4a`
- Preprocessing: resampling, fixed-length normalization
- Feature extraction:
  - MFCC
  - Delta MFCC
  - Delta-Delta MFCC
- Lightweight neural network classifier (non-CNN)
- Optimized for low-latency inference

---

## 🖼️ Image Adversarial Detection

- Image resizing and normalization
- CNN-based detector (ResNet-18)
- Trained on adversarial and natural adversarial images
- Detects hidden instructions and perturbation-based attacks

---

## 🛡️ Risk-Aware Decision Engine

Instead of simple classification, the system applies confidence-based thresholds:
