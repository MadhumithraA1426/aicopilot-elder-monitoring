import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
import os

df = pd.read_csv('dataset/health_monitoring.csv')

# --- Use exactly these column names found in your CSV ---
features = [
    'Heart Rate', 
    'Blood Pressure', 
    'Glucose Levels', 
    'Oxygen Saturation (SpO₂%)', 
    'Alert Triggered (Yes/No)', 
    'Caregiver Notified (Yes/No)'
]
label = 'Alert Triggered (Yes/No)'  # binary label, 1=alert, 0=normal

# --- Clean any NaN ---
df = df.dropna(subset=features)

# --- Convert columns ---
for col in ['Alert Triggered (Yes/No)', 'Caregiver Notified (Yes/No)']:
    df[col] = df[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})

# Convert Blood Pressure from 'NNN/NNN mmHg' to average integer value
def bp_num(x):
    try:
        return sum([int(s) for s in x.split()[0].split('/')]) // 2
    except:
        return 120
df['Blood Pressure'] = df['Blood Pressure'].apply(bp_num)

X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df[label]

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Train DNN ---
model = MLPClassifier(hidden_layer_sizes=(64,32), random_state=42, max_iter=500)
model.fit(X_train, y_train)

# --- Save ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/elderly_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Training complete, model saved to model/elderly_model.pkl")
print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
