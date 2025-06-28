import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def generate_data():
    np.random.seed(42)
    X = np.random.rand(1000, 4) * 100
    y = (X[:, 0] > 70) | (X[:, 1] > 80) | (X[:, 2] > 90)
    return X, y.astype(int)

X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/final_model.pkl")
