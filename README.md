# 🧱 BrickVision AI — Automated Clay Brick Quality Grading System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-SVM-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-teal.svg)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-Admin%20Dashboard-black.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![Author](https://img.shields.io/badge/Author-Md.%20Amanullah%20Sheikh%20Apon-emerald.svg)](https://github.com/Apon-The-Bot)

An industrial computer vision and machine learning system for non-destructive civil engineering inspection... **BrickVision AI** grades clay bricks (Grade 1 to Grade 4) using multi-spectral color modeling, GLCM texture descriptors, and an Out-of-Distribution (OOD) Mahalanobis distance filter to reject non-brick samples (e.g., sand, gravel, or background noise).

> **Author:** Md. Amanullah Sheikh Apon  
> **Contact:** [mdamanullahsheikhapon@gmail.com](mailto:mdamanullahsheikhapon@gmail.com)  
> **Portfolio:** [https://apon-the-bot.github.io](https://apon-the-bot.github.io)

---

## 🔬 Core Technical Highlights

1. **35 Handcrafted Feature Descriptors:**
   - **Quantized GLCM Texture Metrics (8):** Contrast, Correlation, Energy, Homogeneity calculated across 8-level quantized grayscale matrices (aligned with MATLAB `graycomatrix` standards).
   - **Color Space Moments (9):** RGB, HSV, and BT.601 YCbCr mean, variance, and standard deviations across calibrated channels.
   - **Geometric Shape Descriptors (5):** Area, Perimeter, Eccentricity, Solidity, and Extent.
   - **Uniform Local Binary Patterns (10):** LBP texture spectrum histogram.
   - **Custom Mineral & Density Indices (3):** Mean Hue Distance (MHD_Red), Color-Texture Factor (CTF), and Color Gradient Magnitude (CGM).

2. **Out-of-Distribution (OOD) Protection:**
   - Employs **Mahalanobis Distance** across the 35 feature dimensions. If distance $\ge 60.0$, the image is instantly rejected as an `Invalid Specimen` (e.g. sand, soil, grass) with forced confidence reduction.

3. **Automated Specimen Cropping:**
   - Boundary extraction using OpenCV contours to isolate clay brick specimens from white laboratory backgrounds.

4. **Multi-Platform Web Application:**
   - **Client Diagnostic Web Portal:** Upload brick photos, view predicted Grade (G1/G2/G3/G4), 98%+ confidence score, and civil engineering suitability reports.
   - **Protected Admin Training Dashboard:** Bulk dataset uploader, real-time training progress, evaluation graphs (Accuracy & Loss), and class distribution stats.

---

## 🏗️ Project Architecture

```text
Bricks-Classification/
├── backend/
│   ├── app.py                     # Flask web server & admin dashboard API
│   └── predict.py                 # Feature extraction & SVM prediction engine
├── frontend/
│   ├── admin/                     # Protected training dashboard, reports & logs
│   └── user/                      # Interactive user diagnostic testing interface
├── models/
│   ├── best_model_package.pkl     # Exported scikit-learn model bundle
│   ├── BrickSVMModel.mat          # MATLAB-compatible trained weights
│   ├── training_history.png       # Dual evaluation graphs (Accuracy vs Loss)
│   ├── train_val_report.txt       # Training and validation summary report
│   └── model_training_stats.json  # Class-wise dataset split statistics
├── scripts/
│   └── train_model.py             # Feature extraction & SVM training script
├── app.py                         # Standalone FastAPI inference server
├── index.html                     # Glassmorphism laboratory testing interface
├── run_app.bat                    # One-click FastAPI runner
└── run_brick_app.bat              # One-click Flask Dashboard runner
```

---

## 🚀 Quick Setup & Execution

### 1. Requirements Installation
```bash
git clone https://github.com/Apon-The-Bot/brickvision-quality-grading.git
cd brickvision-quality-grading

pip install flask fastapi uvicorn scikit-learn opencv-python numpy scipy matplotlib pillow
```

### 2. Run the Full Diagnostic & Training Dashboard (Flask)
```bash
python backend/app.py
```
Open `http://127.0.0.1:5001/` for the User Diagnostic UI, or `http://127.0.0.1:5001/admin` for the Protected Training Dashboard (Default password: `admin123`).

### 3. Run Standalone FastAPI Server
```bash
python app.py
```
Open `http://127.0.0.1:8000/` in your browser.

---

## 📄 License
This project is open-source under the MIT License.
