from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model/elderly_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    heart_rate = float(request.form['heart_rate'])
    blood_pressure = float(request.form['blood_pressure'])
    glucose_level = float(request.form['glucose_level'])
    oxygen_saturation = float(request.form['oxygen_saturation'])
    alert_triggered = int(request.form['alert_triggered'])
    caregiver_notified = int(request.form['caregiver_notified'])

    arr = np.array([[heart_rate, blood_pressure, glucose_level, oxygen_saturation, alert_triggered, caregiver_notified]])
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)[0]
    status = "⚠️ ALERT Triggered!" if prediction == 1 else "✅ Normal State"
    return render_template('index.html', prediction_text=status)

if __name__ == '__main__':
    app.run(debug=True)
