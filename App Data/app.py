from flask import Flask, request, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
model = joblib.load('best_model.joblib')
scaler = joblib.load('standard_scaler.joblib')  # Ensure you save your scaler during preprocessing and load it here

# Genre Mapping
genre_mapping = {
    'bhajan': 0, 'bhangra': 1, 'carnatic': 2, 'carnatic vocal': 3, 'chutney': 4, 
    'classic bhangra': 5, 'classic bollywood': 6, 'classic kollywood': 7, 'filmi': 8, 
    'ghazal': 9, 'hare krishna': 10, 'hindustani classical': 11, 'hindustani instrumental': 12, 
    'hip hop': 13, 'indian classical': 14, 'jain bhajan': 15, 'kollywood': 16, 'lata': 17, 
    'mantra': 18, 'modern bollywood': 19, 'odia pop': 20, 'pop': 21, 'punjabi pop': 22, 
    'rap': 23, 'rock': 24, 'sandalwood': 25, 'sufi': 26, 'tamil devotional': 27, 
    'tamil pop': 28, 'tollywood': 29
}

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    # Render the input form page
    return render_template('index.html', genres=genre_mapping.keys())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Feature names as per your dataset
        feature_names = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
            'duration_sec', 'genre_encoded'
        ]

        # Extract inputs
        features = [
            float(request.form['danceability']),
            float(request.form['energy']),
            int(request.form['key']),
            float(request.form['loudness']),
            int(request.form['mode']),
            float(request.form['speechiness']),
            float(request.form['acousticness']),
            float(request.form['instrumentalness']),
            float(request.form['liveness']),
            float(request.form['valence']),
            float(request.form['tempo']),
            float(request.form['duration_sec']),
            genre_mapping[request.form['genre']]  # Map genre to encoded value
        ]

        # Scale the features
        input_features = scaler.transform([features])

        # Predict popularity
        prediction = model.predict(input_features)

        # Return the prediction result
        return render_template(
            'index.html',
            genres=genre_mapping.keys(),
            prediction_text=f"Predicted Popularity: {prediction[0]:.2f}"
        )
    except Exception as e:
        return render_template(
            'index.html',
            genres=genre_mapping.keys(),
            error_text=f"Error: {str(e)}"
        )

@app.route('/features')
def features():
    # Render the features information page
    feature_info = {
        "danceability": "Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.",
        "energy": "Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.",
        "key": "The key the track is in. Integers map to pitches using standard Pitch Class notation.",
        "loudness": "The overall loudness of a track in decibels (dB).",
        "mode": "The modality of the track: 1 for major, 0 for minor.",
        "speechiness": "Speechiness detects the presence of spoken words in a track.",
        "acousticness": "A confidence measure from 0.0 to 1.0 of whether the track is acoustic.",
        "instrumentalness": "Predicts whether a track contains no vocals.",
        "liveness": "Detects the presence of an audience in the recording.",
        "valence": "A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.",
        "tempo": "The overall estimated tempo of a track in beats per minute (BPM).",
        "duration_sec": "The duration of the track in seconds.",
        "genre": "The genre of the track, represented as a category mapped to an encoded value."
    }
    return render_template('features.html', feature_info=feature_info)

if __name__ == '__main__':
    app.run(debug=True)
