# Song Popularity Prediction

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

### 3. Hyperparameter Tuning
- Use predefined hyperparameters for Random Forest:
  - `n_estimators=200`
  - `max_depth=10`
  - `min_samples_split=2`
  - `min_samples_leaf=4`
- Use predefined hyperparameters for Gradient Boosting:
  - `subsample=1.0`
  - `n_estimators=100`
  - `max_depth=5`
  - `learning_rate=0.1`

### 4. Model Evaluation
Evaluate models using:
- **Mean Squared Error (MSE)**: Measures prediction errors.
- **R-Squared (R²)**: Proportion of variance explained by the model.

### 5. Save the Best Model
- Save the best-performing model using `joblib` for future use.

## Results
- **Random Forest** achieved the best performance with:
  - Train R²: 0.81
  - Test R²: 0.76
- The model was saved as `best_model.joblib`.

## Usage
1. Clone the repository and set up the environment.
2. Run the preprocessing script to prepare the dataset.
3. Train the models and evaluate performance.
4. Use the saved model for predictions:
   ```python
   import joblib
   model = joblib.load('best_model.joblib')
   sample_input = [[0.5, 0.7, 5, -10.0, 1, 0.05, 0.3, 0.0, 0.15, 0.8, 120.0, 300000, 4]]
   prediction = model.predict(scaler.transform(sample_input))
   print("Predicted Popularity:", prediction[0])
   ```

## Future Enhancements
- Include additional features like lyrics-based sentiment analysis.
- Experiment with deep learning models such as neural networks.
- Deploy the model as an API for real-time predictions.

## Author
This project was developed as part of a data science workflow for predicting song popularity. Contributions and feedback are welcome!

