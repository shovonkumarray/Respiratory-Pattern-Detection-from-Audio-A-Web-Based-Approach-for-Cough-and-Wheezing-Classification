from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import io
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Mock training data and model (simulated for demo)
def create_mock_model():
    # Simulate feature extraction and training data
    X = np.random.rand(100, 13)  # 100 samples, 13 MFCC features
    y = np.array(['normal'] * 40 + ['coughing'] * 30 + ['wheezing'] * 30)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Extract audio features
def extract_features(file):
    try:
        audio, sr = librosa.load(io.BytesIO(file.read()), sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        raise Exception(f"Audio processing error: {str(e)}")

# Mock classification
model = create_mock_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if not file.filename.lower().endswith(('.wav', '.mp3')):
            return jsonify({'error': 'Unsupported file format'}), 400

        features = extract_features(file)
        prediction = model.predict([features])[0]
        confidence = np.max(model.predict_proba([features])) * 100

        return jsonify({
            'pattern': prediction,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)