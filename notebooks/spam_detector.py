import pandas as pd
import re
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import os

# Constants
MODEL_PATH = 'spam_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
DATA_PATH = r'C:\Users\Krishna\Desktop\Spam SMS detection\data\spam.csv'

def load_data(filepath):
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv(filepath, encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def train_model(X_train, y_train):
    """Train and return the model and vectorizer"""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """Evaluate model performance"""
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)
    
    st.subheader("Model Evaluation")
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.text("\nClassification Report:")
    st.text(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

def save_artifacts(model, vectorizer):
    """Save model and vectorizer to disk"""
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_artifacts():
    """Load model and vectorizer from disk"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        return None, None

def main():
    st.title("🛡️ Spam SMS Detector")
    st.write("""
    A machine learning model to classify SMS messages as spam or ham (not spam).
    """)

    # Sidebar for training control
    st.sidebar.header("Model Training")
    retrain = st.sidebar.checkbox("Retrain model", value=False)

    if retrain:
        st.sidebar.warning("Retraining will overwrite existing model!")
        if st.sidebar.button("Train Model"):
            with st.spinner("Training model..."):
                df = load_data(DATA_PATH)
                if df is not None:
                    df['message'] = df['message'].apply(clean_text)
                    X_train, X_test, y_train, y_test = train_test_split(
                        df['message'], df['label'], test_size=0.2, random_state=42)
                    
                    model, vectorizer = train_model(X_train, y_train)
                    save_artifacts(model, vectorizer)
                    evaluate_model(model, vectorizer, X_test, y_test)
                    st.success("Model trained and saved successfully!")
    else:
        model, vectorizer = load_artifacts()
        if model is not None:
            st.success("Loaded pre-trained model successfully!")

    # Prediction interface
    st.header("🔍 Try the Spam Detector")
    user_input = st.text_area("Enter an SMS message to check if it's spam:")

    if st.button("Predict"):
        if 'model' not in locals() or model is None:
            st.warning("Please train or load a model first!")
        elif not user_input:
            st.warning("Please enter a message to predict!")
        else:
            cleaned_input = clean_text(user_input)
            input_vec = vectorizer.transform([cleaned_input])
            prediction = model.predict(input_vec)[0]
            probability = model.predict_proba(input_vec)[0][1]

            if prediction == 1:
                st.error(f"🚨 Spam detected! (confidence: {probability:.2%})")
            else:
                st.success(f"✅ Ham (not spam) (confidence: {1-probability:.2%})")

            st.write("""
            **Interpretation:**
            - **Ham (0):** Legitimate message
            - **Spam (1):** Unwanted promotional/malicious message
            """)

if __name__ == "__main__":
    main()