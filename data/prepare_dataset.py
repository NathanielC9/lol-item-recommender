import pandas as pd
import os

DATA = "data/"
OUT = "data/decision_points.csv"

EXCLUDE = {
    # Consumables / trinkets / wards
    "Health Potion", "Total Biscuit of Everlasting Will",
    "Stealth Ward", "Control Ward", "Oracle Lens", "Farsight Alteration",
    "Elixir of Iron", "Elixir of Sorcery", "Elixir of Wrath",

    # Starter items
    "Doran's Shield", "Doran's Blade", "Doran's Ring",
    "Dark Seal", "Cull", "World Atlas",

    # Jungle starters
    "Gustwalker Hatchling", "Mosstomper Seedling", "Scorchclaw Pup",

    # Boots / basic boots
    "Boots", "Slightly Magical Footwear",

    # Common components
    "Long Sword", "Dagger", "Amplifying Tome", "Ruby Crystal",
    "Cloth Armor", "Null-Magic Mantle", "Sapphire Crystal",
    "Rejuvenation Bead", "Faerie Charm", "Pickaxe", "B. F. Sword",
    "Needlessly Large Rod", "Giant's Belt", "Blasting Wand",
    "Tear of the Goddess", "Sheen", "Kindlegem", "Caulfield's Warhammer",
    "Serrated Dirk", "Recurve Bow", "Vampiric Scepter", "Negatron Cloak",
    "Chain Vest", "Aether Wisp", "Fiendish Codex", "Hextech Alternator",
    "Noonquiver", "Kircheis Shard", "Hearthbound Axe",
    "Winged Moonplate", "Glowing Mote", "Rectrix",

    # Small upgraded components
    "Executioner's Calling", "Bramble Vest", "Oblivion Orb",
    "Spectre's Cowl", "Warden's Mail", "Rageknife",
    "Last Whisper", "Zeal", "Bami's Cinder",
    "Lost Chapter", "Phage", "Tunneler", "Ironspike Whip"
}

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
    rank_tbl = pd.read_csv(DATA + "RankTbl.csv")

    print("Building lookups...")
    champ_lookup = dict(zip(champion_tbl["ChampionId"], champion_tbl["ChampionName"]))
    item_lookup = dict(zip(item_tbl["ItemID"], item_tbl["ItemName"]))
    duration_lookup = dict(zip(match_tbl["MatchId"], match_tbl["GameDuration"]))

    print("QueueType values:")
    print(match_tbl["QueueType"].value_counts(dropna=False).head(30))

    # TEMP: use all matches until we know exact ranked queue labels
    ranked_match_ids = set(match_tbl["MatchId"])
    print(f"Matches used: {len(ranked_match_ids)}")

    print("Merging tables...")
    merged = match_stats.merge(
        summoner_match,
        left_on="SummonerMatchFk",
        right_on="SummonerMatchId",
        how="left"
    )

    merged = merged[merged["MatchFk"].isin(ranked_match_ids)]
    print(f"Rows after ranked filter: {len(merged)}")

    merged["GameDuration"] = merged["MatchFk"].map(duration_lookup)
    merged["time_bucket"] = merged["GameDuration"].apply(
        lambda x: time_bucket(x) if pd.notna(x) else "mid"
    )
    merged["champion"] = merged["ChampionFk"].map(champ_lookup)

    merged = merged.merge(
        team_match,
        left_on="MatchFk",
        right_on="MatchFk",
        how="left"
    )

    print("Building decision points...")
    rows = []
    for _, r in merged.iterrows():
        items = []
        for col in ["item1", "item2", "item3", "item4", "item5", "item6"]:
            val = r.get(col)
            if pd.notna(val) and int(val) != 0:
                item_name = item_lookup.get(int(val), str(int(val)))
                items.append(item_name)

        items = [i for i in items if i not in EXCLUDE]
        if not items:
            continue

        player_champ = r.get("champion")
        if pd.isna(player_champ):
            continue

        blue_champs = [champ_lookup.get(r.get(f"B{i}Champ"), "None") for i in range(1, 6)]
        red_champs = [champ_lookup.get(r.get(f"R{i}Champ"), "None") for i in range(1, 6)]

        if player_champ in blue_champs:
            enemy_champs = red_champs
        else:
            enemy_champs = blue_champs

        kda = f"{int(r.get('kills', 0))}/{int(r.get('deaths', 0))}/{int(r.get('assists', 0))}"
        gold = float(r.get("TotalGold", 0))
        win = int(r.get("Win", 0))
        lane = str(r.get("Lane", "NONE"))
        cs = float(r.get("MinionsKilled", 0))
        vision = float(r.get("visionScore", 0))

        for item in items:
            rows.append({
                "champion": player_champ,
                "lane": lane,
                "time_bucket": r["time_bucket"],
                "kda": kda,
                "gold": gold,
                "win": win,
                "cs": cs,
                "vision": vision,
                "enemy_1": enemy_champs[0] if len(enemy_champs) > 0 else "None",
                "enemy_2": enemy_champs[1] if len(enemy_champs) > 1 else "None",
                "enemy_3": enemy_champs[2] if len(enemy_champs) > 2 else "None",
                "enemy_4": enemy_champs[3] if len(enemy_champs) > 3 else "None",
                "enemy_5": enemy_champs[4] if len(enemy_champs) > 4 else "None",
                "label_item": item
            })

    print(f"Built {len(rows)} rows before balancing...")
    out = pd.DataFrame(rows)

    balanced = []
    for item_name, group in out.groupby("label_item"):
        balanced.append(group.sample(min(len(group), 15000), random_state=42))

    out = pd.concat(balanced).reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out)} rows to {OUT}")
    print("Columns:", out.columns.tolist())

if __name__ == "__main__":
    main()