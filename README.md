
# AI Copilot for Elderly Healthcare Monitoring 🧠💊

## Overview
A deep learning–based monitoring system that detects inactivity, abnormal movement, and triggers alerts for elderly individuals.

## Features
- DNN model for activity detection
- Flask web interface
- Real-time alert simulation
- Deployable on Render/Heroku

## Dataset
- UCI "Activity Recognition with Healthy Older People" Dataset
- Alternative: "AI for Elderly Care and Support" from Kaggle
- A small `sample.csv` is provided for quick testing only.


## How to Run
- pip install -r requirements.txt
- python model/train_model.py
- python app.py


## Deployment
Deploy using Flask + Render or Heroku. Add requirement.txt and Procfile for deployment.

---

## Future Upgrades
- Real sensor data from accelerometers
- Integration with Twilio email/SMS alerts
- Dashboard for caregivers

---

### Recommended References
- DeepCareX AI Monitoring System [web:2]
- AI for Elderly Care Dataset [web:1]
- UCI Elderly Activity Dataset [web:12]
- Flask Deployment Tutorials [web:7][web:16][web:19]
- Fall Risk DNN Models [web:5]

---

### Test the app lively
- https://aicopilot-elder-monitoring.onrender.com/

