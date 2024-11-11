import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 100

# Input features
bp_systolic = np.random.randint(90, 150, size=n_samples)
bp_diastolic = np.random.randint(60, 100, size=n_samples)
stress_level = np.random.randint(0, 41, size=n_samples)
room_temp = np.random.uniform(15, 30, size=n_samples)
humidity = np.random.uniform(30, 70, size=n_samples)
heart_rate = np.random.randint(60, 120, size=n_samples)
light_intensity = np.random.randint(0, 256, size=(n_samples, 3))  # RGB values
air_quality = np.random.randint(0, 501, size=n_samples)

# Output features (target)
optimal_temp = np.random.uniform(18, 24, size=n_samples)
optimal_pressure = np.random.uniform(1, 2, size=n_samples)  # Example pressure scale
optimal_sound_freq = np.random.uniform(0.5, 8, size=n_samples)  # Delta to Theta waves
light_red = np.random.randint(0, 256, size=n_samples)
light_blue = np.random.randint(0, 256, size=n_samples)
light_green = np.random.randint(0, 256, size=n_samples)
light_yellow = np.random.randint(0, 256, size=n_samples)
light_orange = np.random.randint(0, 256, size=n_samples)

# Creating a DataFrame
df = pd.DataFrame({
    'bp_systolic': bp_systolic,
    'bp_diastolic': bp_diastolic,
    'stress_level': stress_level,
    'room_temp': room_temp,
    'humidity': humidity,
    'heart_rate': heart_rate,
    'light_red_input': light_intensity[:, 0],
    'light_blue_input': light_intensity[:, 1],
    'light_green_input': light_intensity[:, 2],
    'air_quality': air_quality,
    'optimal_temp': optimal_temp,
    'optimal_pressure': optimal_pressure,
    'optimal_sound_freq': optimal_sound_freq,
    'light_red': light_red,
    'light_blue': light_blue,
    'light_green': light_green,
    'light_yellow': light_yellow,
    'light_orange': light_orange
})

# Save to CSV for further use
df.to_csv('sleep_conditions_data.csv', index=False)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define input features and target variables
X = df[['bp_systolic', 'bp_diastolic', 'stress_level', 'room_temp', 'humidity', 'heart_rate',
        'light_red_input', 'light_blue_input', 'light_green_input', 'air_quality']]
y = df[['optimal_temp', 'optimal_pressure', 'optimal_sound_freq',
        'light_red', 'light_blue', 'light_green', 'light_yellow', 'light_orange']]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')


import joblib

# Save the trained model to a file
joblib.dump(model, 'sleep_model.pkl')

from flask import Flask, request, jsonify,render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('sleep_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request (JSON format)
    data = request.get_json()

    # Prepare the input data for prediction
    input_features = np.array([[
        data['bp_systolic'], data['bp_diastolic'], data['stress_level'], data['room_temp'],
        data['humidity'], data['heart_rate'], data['light_red_input'],
        data['light_blue_input'], data['light_green_input'], data['air_quality']
    ]])

    # Make prediction using the loaded model
    prediction = model.predict(input_features)

    # Return the predictions as a JSON response
    return jsonify({
        'optimal_temp': prediction[0][0],
        'optimal_pressure': prediction[0][1],
        'optimal_sound_freq': prediction[0][2],
        'light_red': prediction[0][3],
        'light_blue': prediction[0][4],
        'light_green': prediction[0][5],
        'light_yellow': prediction[0][6],
        'light_orange': prediction[0][7]
    })



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

