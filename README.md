# AI-Based Steel Surface Defect Detection

An end-to-end computer vision system for automated detection and localization of surface defects in steel using YOLO11 object detection.

## Project Overview

Surface defects in steel can affect product quality, structural integrity, and manufacturing efficiency. Manual inspection can be time-consuming and difficult to scale consistently.

This project develops an AI-based inspection pipeline that detects and localizes six categories of steel surface defects from images:

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

The project covers the complete computer vision workflow, from dataset preparation and model training to quantitative evaluation, error analysis, model comparison, and inference.

---

## Key Highlights

- Built a YOLO11-based object detection pipeline for industrial surface inspection
- Trained and evaluated on **1,800 steel surface images**
- Detects **6 defect categories**
- Used a dedicated **train / validation / test** split
- Trained locally using **Apple M4 GPU acceleration through PyTorch MPS**
- Performed class-wise performance analysis
- Investigated the most challenging defect category, **crazing**
- Compared baseline and final model performance
- Improved crazing recall from **29.3% to 43.7%**
- Achieved **78.4% precision and 76.4% mAP@50** on the held-out test set
- Generated visual predictions across the complete test set

---

# Dataset

The project uses the **NEU-DET steel surface defect dataset** converted into YOLO-compatible format.

### Dataset Distribution

| Split | Images |
|---|---:|
| Training | 1,260 |
| Validation | 360 |
| Testing | 180 |
| **Total** | **1,800** |

The test set was kept separate from model training and was used for the final performance evaluation.

---

# Defect Categories

| Class ID | Defect Type |
|---:|---|
| 0 | Crazing |
| 1 | Inclusion |
| 2 | Patches |
| 3 | Pitted Surface |
| 4 | Rolled-in Scale |
| 5 | Scratches |

The model performs object detection, meaning it predicts both the **defect category** and its **location through bounding boxes**.

---

# Model Architecture

The project uses **YOLO11n**, a lightweight YOLO object detection architecture suitable for efficient computer vision inference.

### Model Configuration

| Component | Configuration |
|---|---|
| Model | YOLO11n |
| Parameters | 2.58M |
| Input Resolution | 640 × 640 |
| Training Epochs | 100 |
| Framework | Ultralytics |
| Deep Learning Framework | PyTorch |
| Training Hardware | Apple M4 |
| Acceleration | Apple MPS |

The lightweight architecture provides a practical balance between detection performance and computational efficiency.

---

# Development Pipeline

The project was developed through the following workflow:

```text
NEU-DET Dataset
      ↓
Dataset Conversion & Validation
      ↓
YOLO Annotation Verification
      ↓
Train / Validation / Test Split
      ↓
YOLO11n Baseline Training
      ↓
Validation & Test Evaluation
      ↓
Class-wise Error Analysis
      ↓
Crazing Analysis
      ↓
Final Model Training
      ↓
Held-out Test Evaluation
      ↓
Prediction Visualization
      ↓
Final Model