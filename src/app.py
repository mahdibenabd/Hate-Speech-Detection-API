from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import logging
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

model_path = os.getenv('MODEL_PATH', 'src/hate_speech_model.pkl')
vectorizer_path = os.getenv('VECTORIZER_PATH', 'src/vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        message = data['message']
        message_vectorized = vectorizer.transform([message])
        
        prediction = model.predict(message_vectorized)
        
        return jsonify({'is_hate_speech': bool(prediction[0])})
    
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({'error': 'An error occurred while processing the request'}), 500

if __name__ == '__main__':
    app.run(debug=True) 