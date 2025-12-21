# AI vs Real Image Detector

![Python](https://img.shields.io/badge/python-3.13-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.9.1-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A PyTorch-based project that classifies images as **REAL** or **FAKE** (AI-generated) using a Convolutional Neural Network (CNN). This project demonstrates practical computer vision skills and deep learning model deployment.

---

## Project Overview

This project detects whether an image is real or AI-generated. Key functionalities:

- Loads and preprocesses real and fake images.
- Trains a CNN model on your dataset.
- Saves trained model weights for inference.
- Provides a script to predict new images.

---

## Project Structure
```
ai-detection/
├── data/ 
├── models/ 
├── src/
│ ├── model.py
│ ├── train.py
│ ├── predict.py
│ └── utils.py
├── tests/ 
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```