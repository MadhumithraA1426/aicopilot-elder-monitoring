from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    heart_rate = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    glucose = db.Column(db.Float)
    oxygen = db.Column(db.Float)
    alert_triggered = db.Column(db.Integer)
    caregiver_notified = db.Column(db.Integer)
    prediction = db.Column(db.String(32))
    timestamp = db.Column(db.DateTime, default=db.func.now())

    user = db.relationship('User', backref='predictions')
