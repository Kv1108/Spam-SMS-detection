# README.md

This repository contains multiple Machine Learning projects developed during my internship. Each folder is a self-contained project with scripts, models, notebooks, and deployment tools.

---

## 🔍 Project 3: SMS Spam Detection (Advanced NLP + Ensemble Models)

A production-ready spam detection system using:
- Deep NLP preprocessing
- Structural + linguistic feature engineering
- Ensemble models (Random Forest, XGBoost, Naive Bayes) stacked with Logistic Regression

## 📁 Repository Structure

```
📦 Root Folder
├── Customer-Segmentation/              # Project 1
├── Handwritten-text-generation/        # Project 2
├── Spam-SMS-Detection/                 # Project 3 (this one)
│   ├── data/
│   │   └── spam.csv
│   ├── notebooks/
│   │   ├── spam_detector.py            # Advanced spam detector training script
│   │   └── Spam_SMS_detection.ipynb
│   ├── Snapshots/                      # Screenshots for report/demo
│   ├── templates/
│   │   └── index.html                  # Web frontend
│   ├── app.py                          # Flask app for live predictions
│   ├── spam_model.pkl                  # Model (optional legacy)
│   ├── spam_xgb_model.pkl              # Model (optional legacy)
│   ├── tfidf_vectorizer.pkl            # Vectorizer (optional legacy)
│   ├── spam_detection_model.pkl        # ✅ Final trained ensemble model
│   ├── requirements.txt
│   └── README.md                       # This file
```

---

## 🧠 Model Architecture

- **Text Preprocessing**
  - Lowercasing, URL/phone/email removal, emoji removal
  - Tokenization, lemmatization, stopword removal (NLTK)

- **Features**
  - TF-IDF of message content
  - Structural features (char count, digit count, URLs, keyword matches, etc.)

- **Models**
  - Random Forest (RF)
  - XGBoost (XGB)
  - Multinomial Naive Bayes (NB)
  - Combined using `StackingClassifier` with Logistic Regression as the final estimator

---

## ⚙️ How to Run

### 1. Set up environment
```bash
cd Spam-SMS-Detection
python -m venv venv
venv\Scripts\activate     # On Windows
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python notebooks/spam_detector.py
```

This will:
- Train the full pipeline
- Show performance metrics and feature importance
- Save the model to `spam_detection_model.pkl`

### 3. Launch the Web App
```bash
python app.py
```
Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to use the spam detector.

---

## 📊 Evaluation Metrics
- **Confusion Matrix**
- **Classification Report**
- **ROC-AUC Score**
- **Feature Importance Visualization**

---

## 🖼️ Snapshots
Screenshots from the project are available under the `Snapshots/` folder.

---

## 📦 Requirements
Main libraries:
- `scikit-learn`
- `xgboost`
- `nltk`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `emoji`, `flask`

> Install via: `pip install -r requirements.txt`

---

## 🙋 Author
**Krishna Viradiya**  
Internship Projects - 2025

---

## 📝 License
MIT License - Free to use, modify, and distribute.
