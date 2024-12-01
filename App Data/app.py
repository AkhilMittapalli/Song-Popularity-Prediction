from flask import Flask, render_template, request, redirect, url_for
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
import numpy as np

# Spotify API credentials
client_id = 'd19759ecd1234dacb94b5091a1cb585e'
client_secret = '7954c247fd0c48d596f2fa84f1d2c9b8'

# Authenticate with Spotify
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Load the regression model
with open('best_random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search', methods=['POST'])
def search():
    search_query = request.form['search_query']
    results = sp.search(q=search_query, type='track', limit=5)
    tracks = []
    for item in results['tracks']['items']:
        tracks.append({
            'name': item['name'],
            'artist': ', '.join([artist['name'] for artist in item['artists']]),
            'album': item['album']['name'],
            'release_date': item['album']['release_date'],
            'id': item['id']
        })
    return render_template('search_results.html', tracks=tracks, search_query=search_query)

@app.route('/check_popularity/<track_id>')
def check_popularity(track_id):
    # Fetch audio features for the track
    audio_features = sp.audio_features([track_id])[0]
    if audio_features:
        features = np.array([[audio_features['danceability'], audio_features['energy'],
                              audio_features['key'], audio_features['loudness'],
                              audio_features['mode'], audio_features['speechiness'],
                              audio_features['acousticness'], audio_features['instrumentalness'],
                              audio_features['liveness'], audio_features['valence'],
                              audio_features['tempo'], audio_features['duration_ms'],
                              audio_features['time_signature']]])
        # Predict popularity
        predicted_popularity = model.predict(features)[0]
        return render_template('popularity_result.html', popularity=predicted_popularity, track_id=track_id)
    else:
        return render_template('popularity_result.html', error="Audio features not found for this track.")

@app.route('/predict_custom', methods=['GET', 'POST'])
def predict_custom():
    if request.method == 'POST':
        try:
            # Get input values from the form
            danceability = float(request.form['danceability'])
            energy = float(request.form['energy'])
            key = int(request.form['key'])
            loudness = float(request.form['loudness'])
            mode = int(request.form['mode'])
            speechiness = float(request.form['speechiness'])
            acousticness = float(request.form['acousticness'])
            instrumentalness = float(request.form['instrumentalness'])
            liveness = float(request.form['liveness'])
            valence = float(request.form['valence'])
            tempo = float(request.form['tempo'])
            duration_ms = int(request.form['duration_ms'])
            time_signature = int(request.form['time_signature'])

            # Create feature array
            features = np.array([[danceability, energy, key, loudness, mode,
                                  speechiness, acousticness, instrumentalness,
                                  liveness, valence, tempo, duration_ms, time_signature]])
            
            # Make prediction
            popularity = model.predict(features)[0]
            return render_template('custom_predict.html', prediction_text=f'Predicted Popularity: {popularity:.2f}')
        except Exception as e:
            return render_template('custom_predict.html', prediction_text=f'Error: {e}')
    return render_template('custom_predict.html')

if __name__ == '__main__':
    app.run(debug=True)
