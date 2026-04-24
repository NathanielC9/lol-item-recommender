import pandas as pd
import os

DATA = "data/"
OUT = "data/decision_points.csv"

def time_bucket(seconds):
    if seconds < 1200:
        return "early"
    elif seconds < 2400:
        return "mid"
    return "late"

def main():
    print("Loading tables...")
    match_stats = pd.read_csv(DATA + "MatchStatsTbl.csv")
    match_tbl = pd.read_csv(DATA + "MatchTbl.csv")
    summoner_match = pd.read_csv(DATA + "SummonerMatchTbl.csv")
    champion_tbl = pd.read_csv(DATA + "ChampionTbl.csv")
    item_tbl = pd.read_csv(DATA + "ItemTbl.csv")
    team_match = pd.read_csv(DATA + "TeamMatchTbl.csv")

    print("Building lookups...")
    champ_lookup = dict(zip(champion_tbl["ChampionId"], champion_tbl["ChampionName"]))
    item_lookup = dict(zip(item_tbl["ItemID"], item_tbl["ItemName"]))
    duration_lookup = dict(zip(match_tbl["MatchId"], match_tbl["GameDuration"]))

    print("Merging tables...")
    #link match stats to summoner match to get champion and match FK
    merged = match_stats.merge(
        summoner_match,
        left_on="SummonerMatchFk",
        right_on="SummonerMatchId",
        how="left"
    )

    #add game duration
    merged["GameDuration"] = merged["MatchFk"].map(duration_lookup)
    merged["time_bucket"] = merged["GameDuration"].apply(
        lambda x: time_bucket(x) if pd.notna(x) else "mid"
    )

    #resolve player champion name
    merged["champion"] = merged["ChampionFk"].map(champ_lookup)

    #merge team composition
    merged = merged.merge(
        team_match,
        left_on="MatchFk",
        right_on="MatchFk",
        how="left"
    )

    print("Building decision points...")
    rows = []
    for _, r in merged.iterrows():
        #get items, filter out nulls and zeros
        items = []
        for col in ["item1", "item2", "item3", "item4", "item5", "item6"]:
            val = r.get(col)
            if pd.notna(val) and int(val) != 0:
                item_name = item_lookup.get(int(val), str(int(val)))
                items.append(item_name)

        if not items:
            continue

        #determine enemy team based on which team the player is on
        #TeamMatchTbl has B1-B5 and R1-R5
        #We use EnemyChampionFk to figure out which side player is on
        blue_champs = [
            champ_lookup.get(r.get(f"B{i}Champ"), "None") for i in range(1, 6)
        ]
        red_champs = [
            champ_lookup.get(r.get(f"R{i}Champ"), "None") for i in range(1, 6)
        ]

        player_champ = r.get("champion")
        if player_champ in blue_champs:
            enemy_champs = red_champs
        else:
            enemy_champs = blue_champs

        kda = f"{int(r.get('kills', 0))}/{int(r.get('deaths', 0))}/{int(r.get('assists', 0))}"
        gold = float(r.get("TotalGold", 0))
        win = int(r.get("Win", 0))
        lane = str(r.get("Lane", "NONE"))

        for item in items:
            rows.append({
                "champion": player_champ,
                "lane": lane,
                "time_bucket": r["time_bucket"],
                "kda": kda,
                "gold": gold,
                "win": win,
                "enemy_1": enemy_champs[0] if len(enemy_champs) > 0 else "None",
                "enemy_2": enemy_champs[1] if len(enemy_champs) > 1 else "None",
                "enemy_3": enemy_champs[2] if len(enemy_champs) > 2 else "None",
                "enemy_4": enemy_champs[3] if len(enemy_champs) > 3 else "None",
                "enemy_5": enemy_champs[4] if len(enemy_champs) > 4 else "None",
                "label_item": item
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out)} rows to {OUT}")

if __name__ == "__main__":
    main()