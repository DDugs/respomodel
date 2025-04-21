import os
import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizer import custom_tokenizer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://airespiratory.vercel.app"]}})

# Load configuration from a JSON file
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Config file {CONFIG_PATH} not found.")
    exit(1)

# Define paths from config
MODEL_PATH = Path(config.get("model_path", "multi_output_model.pkl"))
ENCODERS_PATH = Path(config.get("encoders_path", "label_encoders.pkl"))
VECTORIZER_PATH = Path(config.get("vectorizer_path", "tfidf_vectorizer.pkl"))

# Load model, encoders, and vectorizer
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        label_encoders = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        tfidf = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure model, encoders, and vectorizer files exist.")
    exit(1)
except AttributeError as e:
    print(f"Error loading pickled files: {e}. Ensure custom_tokenizer is defined correctly.")
    exit(1)

# Derive features and outputs from config or dataset
DATASET_PATH = Path(config.get("dataset_path", "respiratory_health_dataset.csv"))
if DATASET_PATH.exists():
    df = pd.read_csv(DATASET_PATH)
    numeric_features = config.get("numeric_features", [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col not in config.get("output_columns", [])])
    text_feature = config.get("text_feature", next((col for col in df.columns if df[col].dtype == 'object' and col not in config.get("output_columns", [])), None))
    output_columns = config.get("output_columns", [col for col in df.columns if col not in numeric_features and col != text_feature])
else:
    numeric_features = config.get("numeric_features", [])
    text_feature = config.get("text_feature", "Symptoms")
    output_columns = config.get("output_columns", [])

# Validate feature configuration
if not numeric_features or not text_feature or not output_columns:
    print("Error: Invalid feature or output configuration. Check config.json or dataset.")
    exit(1)

# Utility function for error responses
def error_response(message, status_code):
    return jsonify({"error": message}), status_code

@app.route('/info', methods=['GET'])
def get_model_info():
    """Return information about the model"""
    # Check model readiness
    model_ready = all([model is not None, tfidf is not None, label_encoders is not None])
    
    return jsonify({
        "model_name": config.get("model_name", "RespiratoryHealthModel"),
        "model_version": config.get("model_version", "1.0"),
        "status": "ready" if model_ready else "not ready",
        "input_features": numeric_features + [text_feature] if text_feature else numeric_features,
        "output_predictions": output_columns
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions based on input features"""
    if not request.is_json:
        return error_response("Request must be JSON", 400)
    
    data = request.get_json()
    
    # Validate input features
    missing_features = [feat for feat in numeric_features + [text_feature] if feat not in data]
    if missing_features:
        return error_response(f"Missing features: {missing_features}", 400)
    
    try:
        # Validate numeric features
        for feat in numeric_features:
            if not isinstance(data[feat], (int, float)) or data[feat] < 0:
                return error_response(f"Invalid value for {feat}: must be a non-negative number", 400)
        
        # Validate text feature
        if not isinstance(data[text_feature], str) or not data[text_feature].strip():
            return error_response(f"{text_feature} must be a non-empty string", 400)
        
        # Extract numeric features
        numeric_data = [float(data[feat]) for feat in numeric_features]
        numeric_array = np.array([numeric_data])
        
        # Extract and transform text feature using TF-IDF
        text_input = data[text_feature]
        text_features = tfidf.transform([text_input]).toarray()
        
        # Combine numeric and text features
        X = np.hstack([numeric_array, text_features])
        
        # Make predictions
        predictions = model.predict(X)[0]
        
        # Decode predictions
        result = {}
        for i, col in enumerate(output_columns):
            le = label_encoders.get(col)
            if le is None:
                return error_response(f"No label encoder found for {col}", 500)
            predicted_label = le.inverse_transform([predictions[i]])[0]
            result[col] = predicted_label
        
        # Add confidence scores
        probabilities = model.predict_proba(X)
        confidences = {}
        for i, col in enumerate(output_columns):
            prob = probabilities[i][0][predictions[i]]
            confidences[col] = round(float(prob * 100), 1)
        
        # Format response
        response = {
            "predictions": result,
            "confidences": confidences,
            "status": "success"
        }
        
        return jsonify(response)
    
    except ValueError as e:
        return error_response(f"Invalid input data: {str(e)}", 400)
    except Exception as e:
        return error_response(f"Server error: {str(e)}", 500)

if __name__ == '__main__':
    # Load server settings from environment variables
    HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT = int(os.getenv("FLASK_PORT", 5000))
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    app.run(host=HOST, port=PORT, debug=DEBUG)
