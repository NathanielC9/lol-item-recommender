# LoL Item Recommender (CECS 458 Final Project)

## Overview

This project is an AI-powered League of Legends item recommendation system that predicts optimal item builds based on match state.

The system uses a hybrid approach:

Deep Learning Model (MLP)  
→ Role-Based Filtering  
→ Enemy Composition Reranking  
→ Explainable Recommendations  

The goal is to help players make better in-game decisions and reduce information overload.

---

## Dataset

Source:  
Kaggle — *LoL Match History and Summoner Data 80k Matches*  
Author: nathansmallcalder  

The raw dataset contains match, champion, and item data.

We transformed the raw tables into a training dataset:

data/decision_points.csv

### Final Dataset Stats

Rows: 651,798  
Unique champions: 172  
Unique item classes: 240  

### Features

- champion  
- lane  
- time_bucket (early / mid / late)  
- kda  
- gold  
- win  
- cs  
- vision  
- enemy_1 through enemy_5  

### Target

- label_item (recommended item)

---

## Data Cleaning

To improve recommendation quality, we removed noisy item labels such as:

- consumables (potions, elixirs)  
- wards and trinkets  
- starter items  
- jungle starter items  
- boots (basic and completed)  
- item components  
- low-value noisy labels  

This ensures the model learns meaningful completed item recommendations.

---

## Model

We trained a Multilayer Perceptron (MLP) classifier.

Input dimension: 183  
Output classes: 240  

### Performance

Rows: 481,991
Unique item classes: 230
Input dimension: 191
Top-1 accuracy: 30.50%
Top-3 accuracy: 56.69%

---

## System Architecture

User Input (Champion, lane, game state, enemy team)  
↓  
MLP Model → probability distribution over items  
↓  
Game Logic Reranker  
↓  
Top-3 Recommendations  
↓  
Explanation Generator  

---

## Example

### Input

Champion: Ornn  
Lane: Top  
Time: Mid  
Enemy Team: Yone, Yasuo, Master Yi, Morgana, Milio  

### Output

1. Randuin's Omen  
2. Frozen Heart  
3. Thornmail  

### Explanation

Randuin’s Omen is recommended because the enemy team has multiple critical-strike and physical damage threats, making anti-crit armor especially valuable for a tank in mid-game teamfights.

---

## Project Structure

app/                API and inference logic  
data/               dataset and documentation  
models/             training scripts and model  
preprocessing/      encoding logic  
utils/              game logic and item mapping  
saved_models/       trained models (ignored by git)  

---

## Setup

### Install dependencies

pip install -r requirements.txt

### Generate dataset

python data/prepare_dataset.py

### Train model

PYTHONPATH=. python models/train_mlp.py

### Run API

PYTHONPATH=. python -m app.api

---

## Team Roles

- Data Engineer → dataset generation, cleaning, documentation  
- Model Engineer → model training and evaluation  
- Backend → recommendation logic and API  
- Frontend → UI and demo  

---

## Data Engineering Contribution

- Built cleaned recommendation dataset from raw Kaggle tables  
- Removed noisy item labels (components, boots, consumables)  
- Generated 650k+ training examples  
- Balanced item classes  
- Created item metadata for explanations  
- Documented dataset pipeline and statistics  

---

## Future Improvements

- Add live Riot API integration  
- Improve item metadata coverage  
- Add LLM-based explanations  
- Include player inventory and team composition features  
- Support patch-specific meta updates  