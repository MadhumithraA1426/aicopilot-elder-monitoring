from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
import joblib
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, PredictionLog
from forms import LoginForm, RegistrationForm

app = Flask(__name__)
app.config['SECRET_KEY'] = 'e2b17d3c90be854ee1a60beb385b5fbb'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///aicopilot.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

with app.app_context():
    db.create_all()

model = joblib.load('model/elderly_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        new_user = User(username=form.username.data, email=form.email.data)
        new_user.set_password(form.password.data)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    return render_template('login.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    logs = PredictionLog.query.filter_by(user_id=current_user.id).order_by(PredictionLog.timestamp.desc()).all()
    return render_template('dashboard.html', logs=logs)

@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    prediction_text = ""
    if request.method == 'POST':
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
        prediction_text = status

        new_log = PredictionLog(
            user_id=current_user.id,
            heart_rate=heart_rate,
            blood_pressure=blood_pressure,
            glucose=glucose_level,
            oxygen=oxygen_saturation,
            alert_triggered=alert_triggered,
            caregiver_notified=caregiver_notified,
            prediction=status
        )
        db.session.add(new_log)
        db.session.commit()

    return render_template('index.html', prediction_text=prediction_text)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
