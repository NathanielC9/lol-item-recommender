import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
from preprocessing.encode import build_encoders, encode_row

DATA_CSV = "data/decision_points.csv"
SAVE = "saved_models"
os.makedirs(SAVE, exist_ok=True)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

print("Loading data...")
df = pd.read_csv(DATA_CSV)
enc = build_encoders(df, os.path.join(SAVE, "encoders.joblib"))

print("Encoding features...")
X = np.vstack([encode_row(r, enc) for _, r in df.iterrows()]).astype(np.float32)
y = enc["item_le"].transform(df["label_item"].astype(str))

counts = Counter(y)
mask = np.array([counts[label] >= 2 for label in y])
X = X[mask]
y = y[mask]
print(f"After filtering rare items: {len(y)} samples")

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)
joblib.dump(scaler, os.path.join(SAVE, "scaler.joblib"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

in_dim = X.shape[1]
out_dim = len(enc["item_le"].classes_)
print(f"Input dim: {in_dim}, Output classes: {out_dim}")

model = MLP(in_dim, out_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
loss_fn = nn.CrossEntropyLoss()

epochs = 30
batch = 512

print("Training...")
best_acc = 0
patience = 5
patience_counter = 0

for epoch in range(epochs):
    model.train()
    perm = np.random.permutation(len(X_train))
    total_loss = 0
    batches = 0
    for i in range(0, len(perm), batch):
        idx = perm[i:i+batch]
        xb = torch.tensor(X_train[idx])
        yb = torch.tensor(y_train[idx], dtype=torch.long)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batches += 1

    scheduler.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
        acc1 = (preds == y_test).mean()
        acc3 = top_k_accuracy_score(y_test, probs, k=3,
               labels=list(range(out_dim)))
    print(f"Epoch {epoch+1} | Loss: {total_loss/batches:.4f} | Top-1: {acc1:.4f} | Top-3: {acc3:.4f}")

    if acc1 > best_acc:
        best_acc = acc1
        patience_counter = 0
        torch.save(model, os.path.join(SAVE, "mlp_model.pt"))
        joblib.dump({"classes": list(enc["item_le"].classes_)}, os.path.join(SAVE, "mlp_meta.joblib"))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} — best Top-1: {best_acc:.4f}")
            break

print("MLP saved.")