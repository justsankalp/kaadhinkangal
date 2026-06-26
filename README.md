# Real-Time Tamil Fingerspelling Translation System: Image-to-Text Conversion and Speech Synthesis

Official repository for the paper: **"Real-Time Tamil Fingerspelling Translation System: Image-to-Text Conversion and Speech Synthesis"**, published in the *International Conference on Machine Learning, Image Processing, Network Security and Data Sciences (MIND 2024)*, Springer Nature.

[![Paper Link](https://img.shields.io/badge/Springer-Chapter--Link-blue)](https://link.springer.com/chapter/10.1007/978-3-032-14534-5_36)

---

## 📌 Project Overview
This repository contains the core codebase, experimental model pipelines, and evaluation scripts for an end-to-end multi-modal translation system designed for the Tamil fingerspelling community. The framework processes hand-gesture frames depicting Tamil characters, performs robust deep multiclass classification, and routes the localized text predictions to a customized Tamil text-to-speech (TTS) synthesis engine.

By bridging computer vision and speech processing, this architecture aims to mitigate communication barriers and enhance accessibility for Tamil-speaking individuals utilizing fingerspelling systems.

---

## 🏗️ Architecture

The system pipeline consists of three sequential engineering layers:
1. **Data Acquisition & Preprocessing:** Handles image ingestion, standardization, and target augmentation mapping to the **TLFS 23 (Tamil Language Finger Spelling 23)** target schema.
2. **Multiclass Computer Vision Node:** Deep neural network architectures (PyTorch/TensorFlow) configured to accurately classify 247 distinct Tamil fingerspelling characters.
3. **Speech Synthesis Engine:** Receives structured textual outputs from the classification node and outputs high-fidelity audio streams matching spoken Tamil phonetics.

---

## 📂 Repository Structure
```text
├── models/             # Neural network model configurations and weights
├── notebooks/          # Experimental notebooks (Exploratory Data Analysis, training loops)
├── scripts/            # Inference pipelines and utility helpers for TTS mapping
├── requirements.txt    # Python environments and software dependencies
└── README.md
