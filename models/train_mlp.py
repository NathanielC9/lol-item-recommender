import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter

from preprocessing.encode import build_encoders, encode_row
from models.mlp_model import MLP

DATA_CSV = "data/decision_points.csv"
SAVE = "saved_models"

os.makedirs(SAVE, exist_ok=True)

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
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

in_dim = X.shape[1]
out_dim = len(enc["item_le"].classes_)

print(f"Input dim: {in_dim}, Output classes: {out_dim}")

model = MLP(in_dim, out_dim)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5
)

loss_fn = torch.nn.CrossEntropyLoss()

epochs = 50
batch = 512

best_acc = 0
patience = 8
patience_counter = 0

print("Training...")

for epoch in range(epochs):
    model.train()

    perm = np.random.permutation(len(X_train))
    total_loss = 0
    batches = 0

    for i in range(0, len(perm), batch):
        idx = perm[i:i + batch]

        xb = torch.tensor(X_train[idx], dtype=torch.float32)
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
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)

        acc1 = (preds == y_test).mean()
        acc3 = top_k_accuracy_score(
            y_test,
            probs,
            k=3,
            labels=list(range(out_dim))
        )

    print(
        f"Epoch {epoch + 1} | "
        f"Loss: {total_loss / batches:.4f} | "
        f"Top-1: {acc1:.4f} | "
        f"Top-3: {acc3:.4f}"
    )

    if acc3 > best_acc3:
        best_acc3 = acc3
        patience_counter = 0

        torch.save(
            model.state_dict(),
            os.path.join(SAVE, "mlp_model.pt")
        )

        joblib.dump(
            {
                "classes": list(enc["item_le"].classes_),
                "in_dim": in_dim,
                "out_dim": out_dim
            },
            os.path.join(SAVE, "mlp_meta.joblib")
        )

    else:
        patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} — best Top-3: {best_acc3:.4f}")

print("MLP saved.")