import pandas as pd
import os

RAW = "data/raw_data.csv"
OUT = "data/decision_points.csv"

def time_bucket(duration_seconds):
    if duration_seconds < 1200:
        return "early"
    elif duration_seconds < 2400:
        return "mid"
    return "late"

def main():
    df = pd.read_csv(RAW)

    rows = []
    for _, r in df.iterrows():
        items = [r[f"item{i}"] for i in range(6)]
        items = [str(int(i)) for i in items if pd.notna(i) and int(i) != 0]
        if not items:
            continue

        kda = f"{int(r['kills'])}/{int(r['deaths'])}/{int(r['assists'])}"
        gold_diff = int(r['gold_earned']) - int(r['gold_spent'])
        time_b = time_bucket(r['game_duration'])

        for item in items:
            rows.append({
                "champion": r["champion_name"],
                "lane": r["lane"],
                "time_bucket": time_b,
                "kda": kda,
                "gold_diff": gold_diff,
                "win": int(r["win"]),
                "label_item": item
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out)} rows to {OUT}")

if __name__ == "__main__":
    main()