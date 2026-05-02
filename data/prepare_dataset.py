import pandas as pd

DATA = "data/"
OUT = "data/decision_points.csv"


CONSUMABLES = {
    "Health Potion", "Refillable Potion", "Total Biscuit of Everlasting Will",
    "Stealth Ward", "Control Ward", "Oracle Lens", "Farsight Alteration",
    "Elixir of Iron", "Elixir of Sorcery", "Elixir of Wrath"
}

STARTER_ITEMS = {
    "Doran's Shield", "Doran's Blade", "Doran's Ring",
    "Dark Seal", "Cull", "World Atlas",
    "Gustwalker Hatchling", "Mosstomper Seedling", "Scorchclaw Pup"
}

BASIC_BOOTS = {
    "Boots", "Slightly Magical Footwear"
}

COMPLETED_BOOTS = {
    "Berserker's Greaves",
    "Ionian Boots of Lucidity",
    "Mercury's Treads",
    "Plated Steelcaps",
    "Sorcerer's Shoes",
    "Boots of Swiftness",
    "Mobility Boots",
    "Spellslinger's Shoes",
    "Crimson Lucidity"
}

COMPONENT_ITEMS = {
    "Long Sword", "Dagger", "Amplifying Tome", "Ruby Crystal",
    "Cloth Armor", "Null-Magic Mantle", "Sapphire Crystal",
    "Rejuvenation Bead", "Faerie Charm", "Pickaxe", "B. F. Sword",
    "Needlessly Large Rod", "Giant's Belt", "Blasting Wand",
    "Tear of the Goddess", "Sheen", "Kindlegem", "Caulfield's Warhammer",
    "Serrated Dirk", "Recurve Bow", "Vampiric Scepter", "Negatron Cloak",
    "Chain Vest", "Aether Wisp", "Fiendish Codex", "Hextech Alternator",
    "Noonquiver", "Kircheis Shard", "Hearthbound Axe",
    "Winged Moonplate", "Glowing Mote", "Rectrix",
    "Executioner's Calling", "Bramble Vest", "Oblivion Orb",
    "Spectre's Cowl", "Warden's Mail", "Rageknife",
    "Last Whisper", "Zeal", "Bami's Cinder",
    "Lost Chapter", "Phage", "Tunneler", "Ironspike Whip",
    "Cloak of Agility"
}

LOW_PRIORITY_ITEMS = {
    "Steel Sigil"
}

EXCLUDE = (
    CONSUMABLES
    | STARTER_ITEMS
    | BASIC_BOOTS
    | COMPLETED_BOOTS
    | COMPONENT_ITEMS
    | LOW_PRIORITY_ITEMS
)


TANK_CHAMPS = {
    "Ornn", "Malphite", "Sion", "Shen", "Nasus", "Maokai", "Rammus",
    "Sejuani", "Poppy", "TahmKench", "Tahm Kench", "Chogath", "Cho'Gath",
    "DrMundo", "Dr. Mundo", "KSante", "K'Sante"
}

ADC_CHAMPS = {
    "Jinx", "Caitlyn", "Ashe", "Vayne", "Kaisa", "Kai'Sa", "Jhin", "Draven",
    "Ezreal", "Lucian", "MissFortune", "Miss Fortune", "Tristana", "Twitch",
    "Xayah", "Zeri", "Varus", "Sivir", "Samira", "Aphelios", "Kindred", "Graves"
}

MAGE_CHAMPS = {
    "Ahri", "Lux", "Syndra", "Viktor", "Veigar", "Orianna", "Annie",
    "Brand", "Velkoz", "Vel'Koz", "Xerath", "Ziggs", "Malzahar",
    "Cassiopeia", "Morgana", "Katarina", "Diana", "Ekko", "Fizz",
    "Anivia"
}

CRIT_THREATS = {
    "Yasuo", "Yone", "Jinx", "Caitlyn", "Tryndamere", "Tristana",
    "Draven", "Jhin", "Vayne", "Xayah", "Samira", "Aphelios",
    "MasterYi", "Master Yi"
}

AP_THREATS = {
    "Lux", "Ahri", "Syndra", "Viktor", "Veigar", "Brand", "Ziggs",
    "Xerath", "Katarina", "Diana", "Ekko", "Fizz", "Anivia",
    "Cassiopeia", "Morgana", "Malzahar"
}

AD_THREATS = {
    "Yasuo", "Yone", "Jinx", "Caitlyn", "Tryndamere", "Zed", "Talon",
    "Draven", "Jhin", "Vayne", "Darius", "Riven", "Jayce", "Fiora",
    "MasterYi", "Master Yi"
}

ITEM_PRIORITY = {
    # High-impact carry items
    "Infinity Edge": 5,
    "Kraken Slayer": 5,
    "Lord Dominik's Regards": 5,
    "Mortal Reminder": 5,
    "Rabadon's Deathcap": 5,
    "Void Staff": 5,
    "Liandry's Torment": 5,
    "Randuin's Omen": 5,
    "Frozen Heart": 5,
    "Thornmail": 5,
    "Jak'Sho, The Protean": 5,
    "Kaenic Rookern": 5,
    "Force of Nature": 5,

    # Strong completed items
    "Bloodthirster": 4,
    "Guardian Angel": 4,
    "Blade of The Ruined King": 4,
    "Black Cleaver": 4,
    "Trinity Force": 4,
    "Death's Dance": 4,
    "Sterak's Gage": 4,
    "Serylda's Grudge": 4,
    "Shadowflame": 4,
    "Luden's Companion": 4,
    "Zhonya's Hourglass": 4,
    "Stormsurge": 4,
    "Riftmaker": 4,
    "Spirit Visage": 4,
    "Unending Despair": 4,
    "Heartsteel": 4,

    # Medium-priority items
    "The Collector": 3,
    "Rapid Firecannon": 3,
    "Runaan's Hurricane": 3,
    "Phantom Dancer": 3,
    "Immortal Shieldbow": 3,
    "Sundered Sky": 3,
    "Spear of Shojin": 3,
    "Eclipse": 3,
    "Blackfire Torch": 3,
    "Malignance": 3,
    "Rylai's Crystal Scepter": 3,

    # Lower-priority but still valid completed items
    "Statikk Shiv": 2,
}


def choose_best_label_item(items):
    """
    Selects one recommendation label from the cleaned inventory.

    Instead of using the last item in inventory, this picks the item with
    the highest general recommendation value. This reduces noisy labels
    without using champion-specific hardcoding.
    """
    return max(items, key=lambda item: ITEM_PRIORITY.get(item, 3))

def get_champion_role(champion):
    if champion in TANK_CHAMPS:
        return "tank"
    if champion in ADC_CHAMPS:
        return "adc"
    if champion in MAGE_CHAMPS:
        return "mage"
    return "unknown"


def time_bucket(seconds):
    if seconds < 1200:
        return "early"
    elif seconds < 2400:
        return "mid"
    return "late"


def count_enemy_tags(enemy_champs):
    enemy_ad_count = sum(1 for champ in enemy_champs if champ in AD_THREATS)
    enemy_ap_count = sum(1 for champ in enemy_champs if champ in AP_THREATS)
    enemy_crit_count = sum(1 for champ in enemy_champs if champ in CRIT_THREATS)

    return enemy_ad_count, enemy_ap_count, enemy_crit_count


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

    ranked_match_ids = set(match_tbl["MatchId"])

    print("Merging tables...")

    merged = match_stats.merge(
        summoner_match,
        left_on="SummonerMatchFk",
        right_on="SummonerMatchId",
        how="left"
    )

    merged = merged[merged["MatchFk"].isin(ranked_match_ids)]

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

    print("Building cleaned decision points...")

    rows = []

    for _, r in merged.iterrows():
        items = []

        for col in ["item1", "item2", "item3", "item4", "item5", "item6"]:
            val = r.get(col)

            if pd.notna(val) and int(val) != 0:
                item_name = item_lookup.get(int(val), str(int(val)))
                items.append(item_name)

        items = [item for item in items if item not in EXCLUDE]

        if not items:
            continue

        player_champ = r.get("champion")

        if pd.isna(player_champ):
            continue

        blue_champs = [
            champ_lookup.get(r.get(f"B{i}Champ"), "None")
            for i in range(1, 6)
        ]

        red_champs = [
            champ_lookup.get(r.get(f"R{i}Champ"), "None")
            for i in range(1, 6)
        ]

        if player_champ in blue_champs:
            enemy_champs = red_champs
        else:
            enemy_champs = blue_champs

        enemy_ad_count, enemy_ap_count, enemy_crit_count = count_enemy_tags(enemy_champs)

        kills = int(r.get("kills", 0))
        deaths = int(r.get("deaths", 0))
        assists = int(r.get("assists", 0))
        kda_ratio = (kills + assists) / max(1, deaths)

        gold = float(r.get("TotalGold", 0))
        cs = float(r.get("MinionsKilled", 0))
        vision = float(r.get("visionScore", 0))

        game_duration = float(r.get("GameDuration", 1))
        game_minutes = max(game_duration / 60.0, 1.0)

        gold_per_min = gold / game_minutes
        cs_per_min = cs / game_minutes
        vision_per_min = vision / game_minutes

        label_item = choose_best_label_item(items)

        rows.append({
            "champion": player_champ,
            "role": get_champion_role(player_champ),
            "lane": str(r.get("Lane", "NONE")),
            "time_bucket": r["time_bucket"],

            "kills": kills,
            "deaths": deaths,
            "assists": assists,
            "kda_ratio": round(kda_ratio, 3),

            "gold": gold,
            "gold_per_min": round(gold_per_min, 3),
            "win": int(r.get("Win", 0)),
            "cs": cs,
            "cs_per_min": round(cs_per_min, 3),
            "vision": vision,
            "vision_per_min": round(vision_per_min, 3),

            "enemy_1": enemy_champs[0] if len(enemy_champs) > 0 else "None",
            "enemy_2": enemy_champs[1] if len(enemy_champs) > 1 else "None",
            "enemy_3": enemy_champs[2] if len(enemy_champs) > 2 else "None",
            "enemy_4": enemy_champs[3] if len(enemy_champs) > 3 else "None",
            "enemy_5": enemy_champs[4] if len(enemy_champs) > 4 else "None",

            "enemy_ad_count": enemy_ad_count,
            "enemy_ap_count": enemy_ap_count,
            "enemy_crit_count": enemy_crit_count,

            "label_item": label_item
        })

    print(f"Built {len(rows)} cleaned rows before balancing.")

    out = pd.DataFrame(rows)

    print("Balancing item classes...")

    balanced = []

    for item_name, group in out.groupby("label_item"):
        balanced.append(group.sample(min(len(group), 15000), random_state=42))

    out = pd.concat(balanced).reset_index(drop=True)

    print("\nDataset sanity check")
    print("Rows:", len(out))
    print("Unique champions:", out["champion"].nunique())
    print("Unique roles:", out["role"].nunique())
    print("Unique label items:", out["label_item"].nunique())

    print("\nColumns:")
    print(out.columns.tolist())

    print("\nTop 30 label items:")
    print(out["label_item"].value_counts().head(30))

    bad_remaining = out[out["label_item"].isin(EXCLUDE)]["label_item"].unique()
    print("\nExcluded items still present:", bad_remaining)

    out.to_csv(OUT, index=False)

    print(f"\nSaved cleaned dataset to {OUT}")


if __name__ == "__main__":
    main()