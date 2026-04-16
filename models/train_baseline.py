import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, accuracy_score
from preprocessing.encode import build_encoders, encode_row

DATA = "data/decision_points.csv"
SAVE = "saved_models"
os.makedirs(SAVE, exist_ok=True)

df = pd.read_csv(DATA)
enc = build_encoders(df, os.path.join(SAVE, "encoders.joblib"))

X = np.vstack([encode_row(r, enc) for _, r in df.iterrows()])
y = enc["item_le"].transform(df["label_item"].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)
print("Top-1 Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print("Top-3 Accuracy:", top_k_accuracy_score(y_test, probs, k=3))

joblib.dump(model, os.path.join(SAVE, "rf_model.joblib"))
print("Model saved.")