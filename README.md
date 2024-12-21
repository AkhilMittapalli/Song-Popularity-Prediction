# Song Popularity Prediction for Indian Songs

## Overview
This project aims to predict the popularity of songs based on their features using machine learning models. The dataset includes various attributes like danceability, energy, tempo, and more, extracted from audio data. The project evaluates different regression models to identify the best-performing approach for predicting song popularity.

## Features
The dataset contains the following key features:
- **danceability**: Measure of how suitable a track is for dancing.
- **energy**: Intensity and activity level of the track.
- **key**: Key of the track.
- **loudness**: Overall loudness in decibels.
- **mode**: Mode of the track (major or minor).
- **speechiness**: Presence of spoken words.
- **acousticness**: Confidence measure of whether the track is acoustic.
- **instrumentalness**: Predicts whether a track is instrumental.
- **liveness**: Presence of an audience in the recording.
- **valence**: Musical positiveness of a track.
- **tempo**: Beats per minute.
- **duration_ms**: Track duration in milliseconds.
- **time_signature**: Number of beats per bar.
- **popularity**: Target variable, representing the popularity of the track.

## Objectives
1. Perform exploratory data analysis (EDA) to understand the dataset.
2. Preprocess the data for training, including feature scaling and handling missing values.
3. Evaluate multiple regression models:
   - Linear Regression
   - Random Forest Regressor
   - Gradient Boosting Regressor
4. Tune hyperparameters and select the best model based on performance metrics.
5. Save the best model for future predictions.

## Tools and Libraries
- **Python**: Programming language used for analysis and modeling.
- **Scikit-learn**: For machine learning models and preprocessing.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib & Seaborn**: Data visualization.
- **Joblib**: Model serialization.

## Process
### 1. Data Preparation
- Split the dataset into training and test sets.
- Scale features using `StandardScaler` to normalize numerical values.
- Handle missing values and preprocess categorical features (e.g., genre).

### 2. Model Selection
- Train and evaluate the following models:
  - **Linear Regression**: Provides a baseline for performance.
  - **Random Forest Regressor**: Ensemble method to handle non-linear relationships.
  - **Gradient Boosting Regressor**: Sequential ensemble method for better accuracy.

### 3. Models Evaluated
- **Linear Regression**: A simple baseline model.
- **Random Forest**: Ensemble-based model for robust predictions.
- **Gradient Boosting**: Best-performing model with a test R² of 0.781.

### Model Evaluation Metrics
- **Mean Squared Error (MSE)**
- **R-squared (R²)**
- **Mean Absolute Error (MAE)**
   
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

## Acknowledgments
- **Spotify API**: For providing comprehensive metadata and audio features.
- **Scikit-learn**: For machine learning model development.
- **Flask**: For creating the web application.


