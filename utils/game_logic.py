# utils/game_logic.py

TANK_CHAMPS = {
    "Ornn", "Malphite", "Sion", "Shen", "Nasus", "Maokai", "Rammus",
    "Sejuani", "Poppy", "TahmKench", "Chogath", "DrMundo", "KSante"
}

ADC_CHAMPS = {
    "Jinx", "Caitlyn", "Ashe", "Vayne", "Kaisa", "Jhin", "Draven",
    "Ezreal", "Lucian", "MissFortune", "Tristana", "Twitch", "Xayah",
    "Zeri", "Varus", "Sivir", "Samira", "Aphelios"
}

MAGE_CHAMPS = {
    "Ahri", "Lux", "Syndra", "Viktor", "Veigar", "Orianna", "Annie",
    "Brand", "Velkoz", "Xerath", "Ziggs", "Malzahar", "Cassiopeia"
}

CRIT_THREATS = {
    "Yasuo", "Yone", "Jinx", "Caitlyn", "Tryndamere", "Tristana",
    "Draven", "Jhin", "Vayne", "Xayah", "Samira", "Aphelios"
}

AP_THREATS = {
    "Lux", "Ahri", "Syndra", "Viktor", "Veigar", "Brand", "Ziggs",
    "Xerath", "Katarina", "Diana", "Ekko", "Fizz", "Anivia", "Cassiopeia"
}

AD_THREATS = {
    "Yasuo", "Yone", "Jinx", "Caitlyn", "Tryndamere", "Zed", "Talon",
    "Draven", "Jhin", "Vayne", "Darius", "Riven", "Jayce", "Fiora"
}

STARTER_ITEMS = {
    "Doran's Blade", "Doran's Shield", "Doran's Ring", "Cull", "Dark Seal"
}

TANK_ITEMS = {
    "Randuin's Omen", "Frozen Heart", "Thornmail", "Sunfire Aegis",
    "Hollow Radiance", "Heartsteel", "Jak'Sho, The Protean",
    "Kaenic Rookern", "Force of Nature", "Spirit Visage", "Dead Man's Plate"
}

ADC_ITEMS = {
    "Infinity Edge", "Kraken Slayer", "The Collector", "Runaan's Hurricane",
    "Phantom Dancer", "Lord Dominik's Regards", "Mortal Reminder",
    "Bloodthirster", "Statikk Shiv", "Rapid Firecannon", "Guardian Angel"
}

MAGE_ITEMS = {
    "Rabadon's Deathcap", "Luden's Companion", "Shadowflame",
    "Zhonya's Hourglass", "Void Staff", "Liandry's Torment",
    "Morellonomicon", "Banshee's Veil", "Stormsurge"
}


def get_champion_role(champion):
    if champion in TANK_CHAMPS:
        return "tank"
    if champion in ADC_CHAMPS:
        return "adc"
    if champion in MAGE_CHAMPS:
        return "mage"
    return "unknown"


def analyze_enemy_team(raw_row):
    enemies = [
        raw_row.get("enemy_1", "None"),
        raw_row.get("enemy_2", "None"),
        raw_row.get("enemy_3", "None"),
        raw_row.get("enemy_4", "None"),
        raw_row.get("enemy_5", "None"),
    ]

    crit_count = sum(1 for e in enemies if e in CRIT_THREATS)
    ap_count = sum(1 for e in enemies if e in AP_THREATS)
    ad_count = sum(1 for e in enemies if e in AD_THREATS)

    return {
        "enemies": enemies,
        "crit_count": crit_count,
        "ap_count": ap_count,
        "ad_count": ad_count,
    }


def is_valid_item_for_role(item_name, champion):
    role = get_champion_role(champion)

    if role == "tank":
        return item_name in TANK_ITEMS or item_name not in ADC_ITEMS | MAGE_ITEMS

    if role == "adc":
        return item_name in ADC_ITEMS or item_name not in TANK_ITEMS | MAGE_ITEMS

    if role == "mage":
        return item_name in MAGE_ITEMS or item_name not in ADC_ITEMS | TANK_ITEMS

    return True


def item_bonus(item_name, raw_row):
    champion = raw_row.get("champion", "")
    time_bucket = raw_row.get("time_bucket", "mid")
    threat = analyze_enemy_team(raw_row)
    bonus = 0.0

    if time_bucket in {"mid", "late"} and item_name in STARTER_ITEMS:
        bonus -= 0.50

    if threat["crit_count"] >= 3:
        if item_name == "Randuin's Omen":
            bonus += 0.75
        elif item_name == "Frozen Heart":
            bonus += 0.35
        elif item_name == "Thornmail":
            bonus += 0.20

    if threat["ad_count"] >= 3:
        if item_name in {"Randuin's Omen", "Frozen Heart", "Thornmail", "Sunfire Aegis"}:
            bonus += 0.25

    if threat["ap_count"] >= 3:
        if item_name in {"Kaenic Rookern", "Force of Nature", "Spirit Visage", "Hollow Radiance"}:
            bonus += 0.35

    if champion in TANK_CHAMPS and item_name in TANK_ITEMS:
        bonus += 0.20

    if champion in ADC_CHAMPS and item_name in ADC_ITEMS:
        bonus += 0.20

    if champion in MAGE_CHAMPS and item_name in MAGE_ITEMS:
        bonus += 0.20

    return bonus


def explain_item(item_name, raw_row):
    champion = raw_row.get("champion", "")
    time_bucket = raw_row.get("time_bucket", "mid")
    threat = analyze_enemy_team(raw_row)
    role = get_champion_role(champion)

    reasons = []

    if role == "tank" and item_name in TANK_ITEMS:
        reasons.append("This fits a tank build by improving durability.")

    if role == "adc" and item_name in ADC_ITEMS:
        reasons.append("This fits an ADC build by improving damage output.")

    if role == "mage" and item_name in MAGE_ITEMS:
        reasons.append("This fits a mage build by improving ability power or survivability.")

    if threat["crit_count"] >= 3 and item_name == "Randuin's Omen":
        reasons.append("The enemy team has multiple critical strike threats, so anti-crit armor is highly valuable.")

    if threat["ad_count"] >= 3 and item_name in {"Randuin's Omen", "Frozen Heart", "Thornmail", "Sunfire Aegis"}:
        reasons.append("The enemy team is physical damage heavy, so armor is prioritized.")

    if threat["ap_count"] >= 3 and item_name in {"Kaenic Rookern", "Force of Nature", "Spirit Visage", "Hollow Radiance"}:
        reasons.append("The enemy team is magic damage heavy, so magic resistance is prioritized.")

    if time_bucket == "early":
        reasons.append("Early game recommendations focus on lane survival and efficient first purchases.")
    elif time_bucket == "mid":
        reasons.append("Mid game recommendations focus on core item spikes.")
    else:
        reasons.append("Late game recommendations focus on teamfight impact and survivability.")

    if not reasons:
        reasons.append("Recommended based on learned match patterns from the neural network.")

    return " ".join(reasons)