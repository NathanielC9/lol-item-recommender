# LoL Item Recommender

An AI-powered item recommendation system for League of Legends. Built with Python, PyTorch, and Flask.

Given your current game state (champion, lane, enemy team, KDA, gold), the model predicts the best item to build next.

---

## Setup

### 1. Clone the repo

### 2. Create and activate the environment
conda create -n ailol python=3.11
conda activate ailol

### 3. Install dependencies
pip install -r requirements.txt

---

## Running the project

Make sure you always have `(ailol)` in your terminal before running anything.

### Step 1 — Prepare the dataset
https://www.kaggle.com/datasets/jakubkrasuski/league-of-legends-match-dataset-2025?resource=download
Place the raw Kaggle CSV in `data/` and rename it to `raw_data.csv`, then run:
PYTHONPATH=. python data/prepare_dataset.py

### Step 2 — Train the model
PYTHONPATH=. python models/train_mlp.py

### Step 3 — Start the web app
PYTHONPATH=. python app/api.py

### Step 4 — Open the app
Go to `http://localhost:5001` in your browser.

---

## Important notes for teammates

- Never commit `saved_models/` or `data/raw_data.csv` — these are too large for GitHub
- Always run `git pull` before starting work
- Always activate `conda activate ailol` before running anything


