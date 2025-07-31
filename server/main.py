import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app) # This enables cross-origin requests, allowing the frontend to call the backend.

# --- Download NLTK data (only needs to be done once) ---
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    print("Download complete.")

# --- Define file path for the model ---
MODEL_DIR = 'model'
MODEL_FILENAME = os.path.join(MODEL_DIR, 'fake_news_model.joblib')

# --- Text Preprocessing Function (must be identical to the one in modelTrain.py) ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans the input text by lowercasing, removing punctuation and numbers,
    removing stopwords, and applying stemming.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\d+', '', text)      # remove numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# --- 1. Load the Pre-trained Model at startup ---
try:
    print(f"--- Loading model from {MODEL_FILENAME} ---")
    pipeline = joblib.load(MODEL_FILENAME)
    print("Model loaded successfully.\n")
except FileNotFoundError:
    print(f"!!! ERROR: Model file '{MODEL_FILENAME}' not found. !!!")
    print("!!! Please run modelTrain.py first to create the model.     !!!")
    exit()

# --- 2. Define the Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive a headline and return a prediction.
    """
    if not request.json or 'headline' not in request.json:
        return jsonify({'error': 'Missing headline in request'}), 400

    headline = request.json['headline']
    
    # Preprocess the input headline
    cleaned_headline = clean_text(headline)
    
    # Make prediction using the loaded pipeline
    prediction = pipeline.predict([cleaned_headline])
    raw_score = pipeline.decision_function([cleaned_headline])
    
    predicted_label_index = prediction[0]
    confidence_score = abs(raw_score[0])
    label = "Fake News" if predicted_label_index == 1 else "Real News"
    
    # Return the result as JSON
    result = {
        'label': label,
        'score': f'{confidence_score:.2f}'
    }
    return jsonify(result)

# --- 3. Run the Flask App ---
if __name__ == "__main__":
    # The server will run on http://127.0.0.1:5000
    print("Starting Flask server... Go to frontend.html to use the application.")
    app.run(port=5000, debug=True)

