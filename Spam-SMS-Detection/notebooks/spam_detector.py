import pandas as pd
import numpy as np
import re
import string
import urllib.parse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import hstack, csr_matrix
import pickle
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set random seed for reproducibility
np.random.seed(42)

## ------------------------- Data Loading & Preprocessing ------------------------- ##

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.phone_pattern = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
    def preprocess(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(self.url_pattern, ' ', text)
        
        # Remove emails
        text = re.sub(self.email_pattern, ' ', text)
        
        # Remove phone numbers
        text = re.sub(self.phone_pattern, ' ', text)
        
        # Remove emojis
        text = emoji.replace_emoji(text, replace=' ')
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)

## ------------------------- Feature Engineering ------------------------- ##

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.suspicious_keywords = [
            'free', 'win', 'prize', 'click', 'offer', 'congrats', 'guaranteed', 
            'claim', 'urgent', 'discount', 'winner', 'selected', 'limited', 'exclusive',
            'deal', 'risk-free', 'money-back', 'special promotion', 'act now', 'call now',
            'txt', 'text', 'reply', 'stop', 'sms', 'mobile', 'cash', 'loan', 'winning',
            'selected', 'award', 'bonus', 'credit', 'card', 'bank', 'account', 'password'
        ]
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        features = []
        for text in X:
            text = text.lower()
            # Structural features
            char_count = len(text)
            word_count = len(text.split())
            digit_count = sum(c.isdigit() for c in text)
            upper_count = sum(c.isupper() for c in text)
            special_char_count = sum(not c.isalnum() and not c.isspace() for c in text)
            
            # URL features
            has_url = int(bool(re.search(self.url_pattern, text)))
            url_count = len(re.findall(self.url_pattern, text))
            
            # Spam keyword features
            keyword_count = sum(1 for word in self.suspicious_keywords if word in text)
            
            # SMS-specific features
            contains_stop_cmd = int(any(word in text for word in ['stop', 'unsubscribe', 'cancel']))
            contains_reply_cmd = int(any(word in text for word in ['reply', 'call', 'text']))
            
            features.append([
                char_count, word_count, digit_count, upper_count, special_char_count,
                has_url, url_count, keyword_count, contains_stop_cmd, contains_reply_cmd
            ])
        
        return np.array(features)

## ------------------------- Model Training ------------------------- ##

def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    # Convert labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Preprocess text
    preprocessor = TextPreprocessor()
    df['clean_text'] = df['message'].apply(preprocessor.preprocess)
    
    return df

def train_model(df):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
    
    # Create feature pipeline
    text_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        ))
    ])
    
    # Feature union
    feature_union = FeatureUnion([
        ('text_features', text_pipeline),
        ('structural_features', Pipeline([
            ('extractor', FeatureExtractor()),
            ('scaler', FunctionTransformer(lambda x: csr_matrix(x), accept_sparse=True)
        ]))
    ])
    
    # Base models
    base_models = [
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )),
        ('xgb', XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
        )),
        ('nb', MultinomialNB(alpha=0.1))
    ]
    
    # Stacking ensemble
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=0.1,
            solver='saga',
            penalty='elasticnet',
            l1_ratio=0.5
        ),
        stack_method='predict_proba',
        passthrough=True
    )
    
    # Full pipeline
    pipeline = Pipeline([
        ('features', feature_union),
        ('classifier', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    return pipeline, X_test, y_test

## ------------------------- Evaluation & Visualization ------------------------- ##

def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Feature importance (for tree-based models)
    if hasattr(model.named_steps['classifier'].estimators_[0], 'feature_importances_'):
        rf = model.named_steps['classifier'].estimators_[0]
        importances = rf.feature_importances_
        
        # Get feature names
        tfidf_features = model.named_steps['features'].transformer_list[0][1].named_steps['vectorizer'].get_feature_names_out()
        structural_features = ['char_count', 'word_count', 'digit_count', 'upper_count', 'special_char_count',
                             'has_url', 'url_count', 'keyword_count', 'contains_stop_cmd', 'contains_reply_cmd']
        all_features = list(tfidf_features) + structural_features
        
        # Create importance dataframe
        importance_df = pd.DataFrame({'feature': all_features, 'importance': importances})
        importance_df = importance_df.sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance_df)
        plt.title('Top 20 Important Features')
        plt.tight_layout()
        plt.show()

## ------------------------- Main Execution ------------------------- ##

def main():
    # Load and preprocess data
    data_path = r'C:\Users\Krishna\Desktop\Internship-Indolike\Spam-SMS-Detection\data\spam.csv'
    df = load_and_preprocess_data(data_path)
    
    # Class distribution
    print("\nClass Distribution:")
    print(df['label'].value_counts())
    sns.countplot(x='label', data=df)
    plt.title('Class Distribution')
    plt.show()
    
    # Train model
    print("\nTraining model...")
    model, X_test, y_test = train_model(df)
    
    # Evaluate
    print("\nEvaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Save model
    with open('spam_detection_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved as 'spam_detection_model.pkl'")

if __name__ == '__main__':
    main()