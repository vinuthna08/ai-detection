# AI vs Real Image Detector

A **PyTorch-based project** that classifies images as **REAL** or **FAKE (AI-generated)** using a Convolutional Neural Network (CNN).

---

## Overview
This project demonstrates a simple deep learning pipeline for detecting AI-generated images. It includes:

- Loading and preprocessing **real** and **fake** images.
- Training a CNN model on your dataset.
- Saving trained model weights.
- A prediction script for classifying new images.

---

## Project Structure
```
ai-detection/
├── data/ # Real & Fake images (not committed)
│ ├── real/
│ └── fake/
├── models/ # Trained model saved as ai_detector.pth
├── src/
│ ├── model.py # CNN architecture
│ ├── train.py # Training script
│ ├── predict.py # Prediction script
│ └── utils.py # Dataset & transforms
├── tests/ # Optional test scripts
├── requirements.txt # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```
---

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/vinuthna08/ai-detection.git
cd ai-detection
```
2. **Create and activate a virtual environment:**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```
3. **Install Dependencies**
```
pip install -r requirements.txt
```
## Usage 
1. **Train the model**
Make sure data is structured as :
```go
data/
├── real/
├── fake/
```
Run the training script :
```bash
python -m src.train
```
2. **Predict the new image**
```bash
python -m src.predict <image_path>
```




