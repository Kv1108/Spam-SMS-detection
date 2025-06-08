# 📩 Spam SMS Detector

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/yourusername/reponame/notebooks/spam_detector.py)
![GitHub last commit](https://img.shields.io/github/last-commit/Kv1108/Spam-SMS-detection)
![License](https://img.shields.io/badge/License-MIT-green)

A production-ready spam detection system achieving **96% accuracy**.

## 🎯 Features
- 🚀 **One-click demo** via Streamlit Cloud
- 📊 Model evaluation: Precision/Recall/F1 scores
- 🛠️ Modular code: Easy to retrain/extend
- 📦 Includes pre-trained models (`spam_model.pkl`)

## 🤖 How It Works
1. Text preprocessing (lowercasing, special char removal)
2. TF-IDF vectorization
3. Logistic Regression classification

## To run the web app
1. Open Command Prompt (CMD)
2. Navigate to your project folder:cd folder-path
  3. streamlit run notebooks/spam_detector.py

## 🛠️ Installation
```bash
git clone https://github.com/Kv1108/Spam-SMS-detection.git
cd Spam-SMS-detection
pip install -r requirements.txt
