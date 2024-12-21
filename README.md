# Song Popularity Prediction in Indian Music

## Overview
This project leverages machine learning to predict the popularity of Indian songs using metadata and audio features extracted from Spotify's API. The solution is tailored to the diverse and rich Indian music industry, encompassing genres ranging from Bollywood and classical to regional and devotional music. The project includes data collection, preparation, model training, and deployment of a prediction interface using Flask.

---

## Features

- **Data Collection**: Spotify API integration to collect metadata and audio features for Indian songs.
- **Data Preparation**: Cleaning, encoding, and standardizing features for machine learning.
- **Model Training**: Comparison of multiple machine learning models, including Random Forest and Gradient Boosting, with hyperparameter tuning.
- **Deployment**: A Flask-based web application for user-friendly predictions.
- **Exploratory Data Analysis (EDA)**: Insights into correlations between song features and popularity.

---

## Dataset

- **Size**: 47,188 records
- **Features**: 20 columns including metadata (e.g., track name, artist, genre, year) and audio features (e.g., danceability, energy, tempo).
- **Target Variable**: `popularity` (range: 0-100)
- **Source**: Spotify API

---

## Installation

### Prerequisites
- Python 3.7+
- Spotify API credentials (client ID and secret)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/AkhilMittapalli/song-popularity-prediction.git
   cd song-popularity-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Spotify API credentials by creating a `.env` file:
   ```bash
   SPOTIFY_CLIENT_ID=your_client_id
   SPOTIFY_CLIENT_SECRET=your_client_secret
   ```

---

## Usage

### 1. Data Collection
Run the `spotify_data_collection.py` script to collect song data and save it as a CSV:
```bash
python spotify_data_collection.py
```

### 2. Data Preparation and Model Training
Clean and preprocess the dataset and train the model using the `project.py` script:
```bash
python project.py
```

### 4. Deployment
Run the Flask application to provide song popularity predictions:
```bash
python app.py
```
Access the web application at `http://127.0.0.1:5000/`.

---

## Flask Web Application
The web application includes the following pages:
- **Home**: Input song features to predict popularity.
- **Results**: Displays predicted popularity based on the input features.

---

## Machine Learning Models

### Models Evaluated
- **Linear Regression**: A simple baseline model.
- **Random Forest**: Ensemble-based model for robust predictions.
- **Gradient Boosting**: Best-performing model with a test R² of 0.781.

### Model Evaluation Metrics
- **Mean Squared Error (MSE)**
- **R-squared (R²)**
- **Mean Absolute Error (MAE)**

---

## Results

| Model              | Test R²  | Test MSE | Test MAE |
|--------------------|----------|----------|----------|
| Gradient Boosting  | 0.781    | 94.85    | 6.99     |
| Random Forest      | 0.773    | 98.44    | 7.04     |
| Linear Regression  | 0.268    | 317.37   | 14.37    |

---

## Future Work
- Incorporate lyrics and cultural attributes for deeper insights.
- Expand the dataset to include more years and genres.
- Optimize the web application interface for real-time predictions.

---

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for improvements.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- **Spotify API**: For providing comprehensive metadata and audio features.
- **Scikit-learn**: For machine learning model development.
- **Flask**: For creating the web application.
