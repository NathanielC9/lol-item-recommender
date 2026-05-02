# League of Legends Item Recommender (AI Project)

## Overview

This project builds an AI-powered item recommendation system for League of Legends.
Given a game state (champion, lane, stats, and enemy composition), the model predicts the **best next item to build**.

The system combines:

* A **deep learning model (MLP)** trained on historical match data
* A **data-engineered pipeline** for clean feature extraction
* A **metadata-driven explanation system** for human-readable recommendations

---

## AI / Machine Learning Approach

We train a **multi-class classification model** to predict the optimal item:

* Input: Game state features (191 dimensions)
* Output: Probability distribution over 230 possible items
* Model: Multi-Layer Perceptron (MLP)
* Loss: Cross-entropy
* Optimization: Adam + learning rate scheduler + early stopping

## Key Idea

Instead of hardcoding rules, the model **learns patterns from real matches**, such as:

* When to build anti-heal
* When to build armor vs magic resist
* When to prioritize scaling vs survivability

---

##Final Model Performance

| Metric         | Value     |
| -------------- | --------- |
| Samples        | 481,991   |
| Classes        | 230 items |
| Input Features | 191       |
| Top-1 Accuracy | ~30%      |
| Top-3 Accuracy | ~56%      |

Top-3 is emphasized because recommendation systems often present multiple viable options.

---

## Data Engineering Pipeline

## Dataset Processing

`data/prepare_dataset.py`

Major improvements:

* Removed **low-value/noisy items**:

  * components (Long Sword, Ruby Crystal)
  * consumables (potions, wards)
  * starter items
* Replaced label:

  * Last inventory item
  *  **Highest-priority core item**
* Balanced item classes
* Added engineered features

### Features Used

* Champion + lane + role
* Kills / deaths / assists + KDA ratio
* Gold + gold per minute
* CS + CS per minute
* Vision + vision per minute
* Enemy composition:

  * AD count
  * AP count
  * Crit threats

---

## Explanation System

Instead of generic text, explanations are generated using:

```text
data/item_metadata.csv
```

Each item includes:

* stat_tags
* counter_tags
* effect_tags

This allows explanations like:

> "Thornmail is recommended because it reduces healing and punishes auto attackers, making it strong against physical damage teams with sustain."

---

## How to Run

### 1. Add raw data (not included in repo)

Place the following in `/data`:

* MatchStatsTbl.csv
* SummonerMatchTbl.csv
* TeamMatchTbl.csv
* MatchTbl.csv
* ChampionTbl.csv
* ItemTbl.csv

---

### 2. Build dataset

```bash
python data/prepare_dataset.py
```

---

### 3. Train model

```bash
PYTHONPATH=. python models/train_mlp.py
```

---

### 4. Run API

```bash
PYTHONPATH=. python -m app.api
```

---

## 📁 Project Structure

```text
data/
  prepare_dataset.py
  item_metadata.csv
  DATASET_NOTES.md
  data_summary.txt
  final_training_results.txt

models/
  train_mlp.py
  mlp_model.py

preprocessing/
  encode.py

utils/
  game_logic.py
```

---

## Notes

Not included in Git:

* `data/decision_points.csv`
* `saved_models/`
* `__pycache__/`, `.pyc`

These are generated locally.

---

## 🎯 Future Work

* Improve model architecture (embeddings, deeper network)
* Personalize recommendations per player
* Add real-time in-game integration
* Enhance explanation generation with LLMs

---
