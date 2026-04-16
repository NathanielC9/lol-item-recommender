# League of Legends Item Optimizer

A live item recommender for League of Legends based on training data of Diamond+ games. Utilizes Riot's Live Client Data API to read in-game data in real-time, with the option to import the match details of a current state of a game to be fed into an AI-trained model.

Project layout:

- `src/item_rec_page/`: application package
- `main.py`: thin local launcher so `python main.py ...` still works without installing first
- `data/`: collected training data and Riot item catalog
- `models/`: saved Keras artifacts

What it does:

- Reads active-game data from Riot's local Live Client Data API.
- Pulls ranked ladder and match history data from Riot's official web APIs.
- Builds purchase-event training rows from match timelines.
- Trains a Keras model to predict likely next purchases from champion, role, inventory, and game-state features.
- Runs local live inference from the active game snapshot.

What it intentionally does not do:

- Scrape OP.GG.
- Ship a production in-game overlay.

Riot policy note:

Riot's League policy says approved products may not provide game-session-specific information that was previously unknown to the player and may not dictate player decisions. This repo is therefore structured as a local research/prototyping tool rather than a public real-time assistant. Source: https://developer.riotgames.com/docs/lol?from=20423&from_column=20423

## Install

macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

Windows Command Prompt:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
pip install -e .
```

## Environment

Set a Riot development key before collection:

macOS or Linux:

```bash
export RIOT_API_KEY="RGAPI-..."
export RIOT_PLATFORM="NA1"
export RIOT_REGION="AMERICAS"
```

Windows PowerShell:

```powershell
$env:RIOT_API_KEY="RGAPI-..."
$env:RIOT_PLATFORM="NA1"
$env:RIOT_REGION="AMERICAS"
```

Riot's developer portal notes that development keys deactivate every 24 hours:
https://developer.riotgames.com/docs/portal

## Commands

Monitor the live client and wait until a game becomes available:

```bash
python main.py live
```

One-shot snapshot only:

```bash
python main.py live --once --pretty
```

Collect diamond+ ranked training examples:

```bash
python main.py collect --max-players 120 --matches-per-player 20 --max-matches 600
```

Train the Keras model:

```bash
python main.py train
```

Predict from the current live game:

```bash
python main.py predict-live --role BOTTOM --top-k 5
```

The live predictor expects you to provide your role because the local Live Client API does not reliably expose lane/role information for the active player.

`python main.py live` now behaves like a monitor:

- If League or the Live Client API is not available, it stays running and waits.
- When a game becomes available, it automatically resumes printing live updates.
- When the game ends, it drops back into waiting mode instead of crashing.

Cross-platform note:

- The code uses `pathlib`, `requests`, and a pure Python CLI, so the same commands work on both macOS and Windows once Python and dependencies are installed.
- Riot's local Live Client endpoint is still `https://127.0.0.1:2999/liveclientdata` on both platforms.
