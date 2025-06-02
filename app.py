from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('RandomForest.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    time = float(request.form.get('time', 0))
    discharge = float(request.form.get('discharge', 0))
    season_str = request.form.get('season', '').strip().lower()
    temperature = float(request.form.get('temperature', 0))
    ph = float(request.form.get('ph', 0))

    # Map season string to label-encoded value
    season_map = {'winter': 0, 'spring': 1, 'summer': 2, 'autumn': 3}
    if season_str not in season_map:
        return render_template('index.html', prediction=None, error="Invalid season. Please enter Winter, Spring, Summer, or Autumn.")
    season = season_map[season_str]

    features = [time, discharge, season, temperature, ph]
    prediction = model.predict([features])[0]
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
