from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from keras import Input, Model, callbacks, layers, metrics, models, optimizers
from scipy import sparse
from scipy.special import softmax
from sklearn.model_selection import train_test_split

from item_rec_page.config import Settings
from item_rec_page.dataset import INVENTORY_COLUMNS, normalize_role
from item_rec_page.riot_api import estimate_inventory_value, item_name_from_catalog, load_item_catalog


MODEL_FILENAME = "item_model.keras"
METADATA_FILENAME = "model_metadata.json"
PRIOR_FILENAME = "inventory_item_prior.npz"

NUMERIC_FEATURES = [
    "game_time_seconds",
    "level",
    "current_gold",
    "total_gold",
    "cs",
    "jungle_cs",
    "gold_diff_vs_role",
    "level_diff_vs_role",
    "cs_diff_vs_role",
    "team_gold_diff",
    "team_level_diff",
]


@dataclass(frozen=True)
class PredictionArtifacts:
    metadata: dict
    model: Model
    prior: sparse.csr_matrix
    item_catalog: dict[str, dict] | None


def train_model(dataset_path: Path, model_dir: Path, epochs: int = 30, batch_size: int = 256) -> dict:
    dataframe = pd.read_csv(dataset_path)
    if dataframe.empty:
        raise ValueError(f"No training rows found in {dataset_path}")

    dataframe["role"] = dataframe["role"].map(normalize_role)
    label_counts = dataframe["label_item_id"].astype(str).value_counts()
    frequent_labels = label_counts[label_counts >= 8].index
    dataframe = dataframe[dataframe["label_item_id"].astype(str).isin(frequent_labels)].reset_index(drop=True)
    if dataframe.empty:
        raise ValueError("Training dataset is too small after filtering rare item labels.")

    metadata = build_metadata(dataframe)
    encoded = encode_dataframe(dataframe, metadata)

    indices = np.arange(len(dataframe))
    stratify = encoded["y"] if len(np.unique(encoded["y"])) > 1 else None
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=stratify)

    train_inputs = slice_inputs(encoded["inputs"], train_idx)
    val_inputs = slice_inputs(encoded["inputs"], val_idx)
    y_train = encoded["y"][train_idx]
    y_val = encoded["y"][val_idx]

    model = build_model(metadata)
    history = model.fit(
        train_inputs,
        y_train,
        validation_data=(val_inputs, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_top5", mode="max", patience=4, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
        ],
        verbose=2,
    )

    val_predictions = model.predict(val_inputs, verbose=0)
    top1 = float((val_predictions.argmax(axis=1) == y_val).mean())
    top5 = float(top_k_accuracy(val_predictions, y_val, k=min(5, val_predictions.shape[1])))

    prior = build_inventory_prior(dataframe.iloc[train_idx].reset_index(drop=True), metadata)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / MODEL_FILENAME)
    with (model_dir / METADATA_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    sparse.save_npz(model_dir / PRIOR_FILENAME, prior)

    return {
        "dataset_rows": int(len(dataframe)),
        "num_labels": int(len(metadata["label_vocab"])),
        "num_champions": int(len(metadata["champion_vocab"])),
        "num_roles": int(len(metadata["role_vocab"])),
        "top1_accuracy": round(top1, 4),
        "top5_accuracy": round(top5, 4),
        "epochs_ran": len(history.history["loss"]),
        "model_path": str(model_dir / MODEL_FILENAME),
    }


def build_metadata(dataframe: pd.DataFrame) -> dict:
    champion_vocab = sorted(set(dataframe["champion"].astype(str)))
    role_vocab = sorted(set(dataframe["role"].astype(str)))

    inventory_items = set()
    for column in INVENTORY_COLUMNS:
        inventory_items.update(dataframe[column].astype(str).tolist())
    inventory_items.discard("0")

    label_vocab = sorted(set(dataframe["label_item_id"].astype(str)))

    numeric_means = dataframe[NUMERIC_FEATURES].astype(float).mean().to_dict()
    numeric_stds = dataframe[NUMERIC_FEATURES].astype(float).std().replace(0, 1).fillna(1).to_dict()

    return {
        "numeric_features": NUMERIC_FEATURES,
        "champion_vocab": champion_vocab,
        "role_vocab": role_vocab,
        "item_vocab": ["0"] + sorted(inventory_items | set(label_vocab)),
        "label_vocab": label_vocab,
        "numeric_means": numeric_means,
        "numeric_stds": numeric_stds,
    }


def encode_dataframe(dataframe: pd.DataFrame, metadata: dict) -> dict:
    champion_map = {value: index + 1 for index, value in enumerate(metadata["champion_vocab"])}
    role_map = {value: index + 1 for index, value in enumerate(metadata["role_vocab"])}
    item_map = {value: index for index, value in enumerate(metadata["item_vocab"])}
    label_map = {value: index for index, value in enumerate(metadata["label_vocab"])}

    numeric_frame = dataframe[NUMERIC_FEATURES].astype(float).copy()
    for feature in NUMERIC_FEATURES:
        mean = float(metadata["numeric_means"][feature])
        std = float(metadata["numeric_stds"][feature]) or 1.0
        numeric_frame[feature] = (numeric_frame[feature] - mean) / std

    champion = dataframe["champion"].astype(str).map(lambda value: champion_map.get(value, 0)).to_numpy(dtype=np.int32)
    role = dataframe["role"].astype(str).map(lambda value: role_map.get(value, 0)).to_numpy(dtype=np.int32)
    inventory = np.column_stack(
        [
            dataframe[column].astype(str).map(lambda value: item_map.get(value, 0)).to_numpy(dtype=np.int32)
            for column in INVENTORY_COLUMNS
        ]
    )
    labels = dataframe["label_item_id"].astype(str).map(label_map).to_numpy(dtype=np.int32)

    return {
        "inputs": {
            "champion": champion,
            "role": role,
            "inventory": inventory,
            "numeric": numeric_frame.to_numpy(dtype=np.float32),
        },
        "y": labels,
    }


def build_model(metadata: dict) -> Model:
    champion_input = Input(shape=(1,), dtype="int32", name="champion")
    role_input = Input(shape=(1,), dtype="int32", name="role")
    inventory_input = Input(shape=(6,), dtype="int32", name="inventory")
    numeric_input = Input(shape=(len(NUMERIC_FEATURES),), dtype="float32", name="numeric")

    champion_embedding = layers.Flatten()(layers.Embedding(len(metadata["champion_vocab"]) + 1, 24)(champion_input))
    role_embedding = layers.Flatten()(layers.Embedding(len(metadata["role_vocab"]) + 1, 8)(role_input))
    inventory_embedding = layers.GlobalAveragePooling1D()(layers.Embedding(len(metadata["item_vocab"]), 32)(inventory_input))

    x = layers.Concatenate()([champion_embedding, role_embedding, inventory_embedding, numeric_input])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(len(metadata["label_vocab"]), activation="softmax")(x)

    model = Model(
        inputs={
            "champion": champion_input,
            "role": role_input,
            "inventory": inventory_input,
            "numeric": numeric_input,
        },
        outputs=output,
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "sparse_categorical_accuracy",
            metrics.SparseTopKCategoricalAccuracy(k=min(5, len(metadata["label_vocab"])), name="top5"),
        ],
    )
    return model


def build_inventory_prior(dataframe: pd.DataFrame, metadata: dict) -> sparse.csr_matrix:
    item_map = {value: index for index, value in enumerate(metadata["item_vocab"])}
    label_map = {value: index for index, value in enumerate(metadata["label_vocab"])}

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []

    for _, row in dataframe.iterrows():
        label_index = label_map[str(row["label_item_id"])]
        seen_items = {str(row[column]) for column in INVENTORY_COLUMNS if str(row[column]) != "0"}
        for item_id in seen_items:
            row_indices.append(item_map[item_id])
            col_indices.append(label_index)
            values.append(1.0)

    matrix = sparse.coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(len(metadata["item_vocab"]), len(metadata["label_vocab"])),
    ).tocsr()
    return matrix


def load_prediction_artifacts(model_dir: Path) -> PredictionArtifacts:
    metadata_path = model_dir / METADATA_FILENAME
    model_path = model_dir / MODEL_FILENAME
    prior_path = model_dir / PRIOR_FILENAME

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    settings = Settings.from_env()
    return PredictionArtifacts(
        metadata=metadata,
        model=models.load_model(model_path),
        prior=sparse.load_npz(prior_path),
        item_catalog=load_item_catalog(settings.item_catalog_path),
    )


def predict_live_snapshot(
    snapshot: dict,
    role: str,
    model_dir: Path | None = None,
    top_k: int = 5,
    artifacts: PredictionArtifacts | None = None,
) -> dict:
    if artifacts is None:
        if model_dir is None:
            raise ValueError("model_dir is required when prediction artifacts are not preloaded.")
        artifacts = load_prediction_artifacts(model_dir)

    metadata = artifacts.metadata
    model = artifacts.model
    prior = artifacts.prior
    item_catalog = artifacts.item_catalog

    row = live_snapshot_to_row(snapshot, role, item_catalog)
    encoded = encode_dataframe(pd.DataFrame([row]), metadata)
    model_probs = model.predict(encoded["inputs"], verbose=0)[0]
    prior_probs = inventory_prior_distribution(row, metadata, prior)
    combined = softmax(0.85 * np.log(model_probs + 1e-9) + 0.15 * np.log(prior_probs + 1e-9))

    label_vocab = metadata["label_vocab"]
    top_indices = np.argsort(combined)[::-1][:top_k]
    active_player = snapshot["active_player"]

    return {
        "active_player": {
            "summoner_name": active_player["summoner_name"],
            "champion": active_player["champion_name"],
            "role": normalize_role(role),
            "game_time_seconds": snapshot["game_time_seconds"],
        },
        "recommendations": [
            {
                "item_id": label_vocab[index],
                "item_name": item_name_from_catalog(label_vocab[index], item_catalog),
                "score": round(float(combined[index]), 4),
            }
            for index in top_indices
        ],
        "policy_note": "Use this as a local research aid. Riot's policy restricts public products that dictate real-time decisions during gameplay.",
    }


def live_snapshot_to_row(snapshot: dict, role: str, item_catalog: dict[str, dict] | None) -> dict:
    active_player = snapshot["active_player"]
    players = snapshot["players"]
    active_team = active_player.get("team")

    inventory_ids = [int(item_id) for item_id in active_player.get("item_ids", []) if int(item_id) > 0]
    inventory_value = estimate_inventory_value(inventory_ids, item_catalog)
    current_gold = float(active_player.get("current_gold", 0.0))

    team_gold = 0.0
    enemy_gold = 0.0
    team_level = 0.0
    enemy_level = 0.0
    for player in players:
        gold_value = estimate_inventory_value(player.get("item_ids", []), item_catalog)
        if player.get("team") == active_team:
            team_gold += gold_value
            team_level += float(player.get("level", 0))
        else:
            enemy_gold += gold_value
            enemy_level += float(player.get("level", 0))

    row = {
        "champion": active_player["champion_name"],
        "role": normalize_role(role),
        "game_time_seconds": float(snapshot.get("game_time_seconds", 0.0)),
        "level": float(active_player.get("level", 0)),
        "current_gold": current_gold,
        "total_gold": current_gold + inventory_value,
        "cs": float(active_player.get("scores", {}).get("creep_score", 0)),
        "jungle_cs": 0.0,
        "gold_diff_vs_role": 0.0,
        "level_diff_vs_role": 0.0,
        "cs_diff_vs_role": 0.0,
        "team_gold_diff": float(team_gold - enemy_gold),
        "team_level_diff": float(team_level - enemy_level),
        "label_item_id": "0",
    }

    padded_inventory = inventory_ids[:6]
    while len(padded_inventory) < 6:
        padded_inventory.append(0)
    for column, item_id in zip(INVENTORY_COLUMNS, padded_inventory):
        row[column] = str(item_id)
    return row


def inventory_prior_distribution(row: dict, metadata: dict, prior: sparse.csr_matrix) -> np.ndarray:
    item_map = {value: index for index, value in enumerate(metadata["item_vocab"])}
    indices = [
        item_map.get(str(row[column]), 0)
        for column in INVENTORY_COLUMNS
        if str(row[column]) != "0"
    ]
    if not indices:
        return np.full(len(metadata["label_vocab"]), 1.0 / len(metadata["label_vocab"]), dtype=np.float32)

    scores = np.asarray(prior[indices].sum(axis=0)).ravel()
    if not scores.any():
        return np.full(len(metadata["label_vocab"]), 1.0 / len(metadata["label_vocab"]), dtype=np.float32)

    scores = scores / scores.sum()
    return scores.astype(np.float32)


def slice_inputs(inputs: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[indices] for name, values in inputs.items()}


def top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, k: int) -> float:
    top_indices = np.argsort(probabilities, axis=1)[:, -k:]
    matches = [label in row for label, row in zip(labels, top_indices)]
    return float(np.mean(matches))
