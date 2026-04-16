import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, accuracy_score
from preprocessing.encode import build_encoders, encode_row

DATA = "data/decision_points.csv"
SAVE = "saved_models"
os.makedirs(SAVE, exist_ok=True)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

df = pd.read_csv(DATA)
enc = build_encoders(df, os.path.join(SAVE, "encoders.joblib"))

X = np.vstack([encode_row(r, enc) for _, r in df.iterrows()]).astype(np.float32)
y = enc["item_le"].transform(df["label_item"].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

in_dim = X.shape[1]
out_dim = len(enc["item_le"].classes_)
model = MLP(in_dim, out_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    perm = np.random.permutation(len(X_train))
    for i in range(0, len(perm), 256):
        idx = perm[i:i+256]
        xb = torch.tensor(X_train[idx])
        yb = torch.tensor(y_train[idx], dtype=torch.long)
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
        acc1 = (preds == y_test).mean()
        acc3 = top_k_accuracy_score(y_test, probs, k=3)
    print(f"Epoch {epoch+1} | Top-1: {acc1:.4f} | Top-3: {acc3:.4f}")

torch.save(model, os.path.join(SAVE, "mlp_model.pt"))
joblib.dump({"classes": list(enc["item_le"].classes_)}, os.path.join(SAVE, "mlp_meta.joblib"))
print("MLP saved.")