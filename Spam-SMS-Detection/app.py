from flask import Flask, request, jsonify, render_template
import pickle
import re

app = Flask(__name__)

# Load model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['message']
    cleaned = clean_text(user_input)
    vect_input = vectorizer.transform([cleaned])
    prediction = model.predict(vect_input)[0]
    prob = model.predict_proba(vect_input)[0][1]

    result = {
        'prediction': int(prediction),
        'confidence': round(prob if prediction == 1 else 1 - prob, 2)
    }
    return render_template('index.html', result=result, input_message=user_input)

if __name__ == '__main__':
    app.run(debug=True)
