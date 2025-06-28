import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

X = np.random.rand(1000, 4) * 100
y = (X[:, 0] > 70) | (X[:, 1] > 80) | (X[:, 2] > 90)
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
joblib.dump(model, 'models/final_model.pkl')