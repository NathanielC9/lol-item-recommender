TANK_CHAMPS = {
    "Ornn", "Malphite", "Sion", "Shen", "Nasus", "Maokai", "Rammus",
    "Sejuani", "Poppy", "TahmKench", "Tahm Kench", "Chogath", "Cho'Gath",
    "DrMundo", "Dr. Mundo", "KSante", "K'Sante"
}

ADC_CHAMPS = {
    "Jinx", "Caitlyn", "Ashe", "Vayne", "Kaisa", "Kai'Sa", "Jhin", "Draven",
    "Ezreal", "Lucian", "MissFortune", "Miss Fortune", "Tristana", "Twitch",
    "Xayah", "Zeri", "Varus", "Sivir", "Samira", "Aphelios"
}

JUNGLE_MARKSMAN_CHAMPS = {
    "Kindred", "Graves"
}

MAGE_CHAMPS = {
    "Ahri", "Lux", "Syndra", "Viktor", "Veigar", "Orianna", "Annie",
    "Brand", "Velkoz", "Vel'Koz", "Xerath", "Ziggs", "Malzahar", "Cassiopeia"
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

STARTER_ITEMS = {
    "Doran's Blade", "Doran's Shield", "Doran's Ring", "Cull", "Dark Seal"
}

COMPLETED_BOOTS = {
    "Berserker's Greaves",
    "Plated Steelcaps",
    "Mercury's Treads",
    "Ionian Boots of Lucidity",
    "Sorcerer's Shoes",
    "Boots of Swiftness",
    "Mobility Boots",
    "Spellslinger's Shoes",
    "Crimson Lucidity"
}

EARLY_SCALING_ITEMS = {
    "Heartsteel",
    "Rod of Ages",
    "Manamune",
    "Muramana",
    "Archangel's Staff",
    "Seraph's Embrace"
}

LATE_GAME_POWER_ITEMS = {
    "Rabadon's Deathcap",
    "Infinity Edge",
    "Lord Dominik's Regards",
    "Void Staff",
    "Bloodthirster",
    "Guardian Angel",
    "Jak'Sho, The Protean",
    "Randuin's Omen",
    "Force of Nature",
    "Kaenic Rookern",
    "Frozen Heart",
    "Thornmail"
}

TANK_ITEMS = {
    "Randuin's Omen",
    "Frozen Heart",
    "Thornmail",
    "Sunfire Aegis",
    "Hollow Radiance",
    "Heartsteel",
    "Jak'Sho, The Protean",
    "Kaenic Rookern",
    "Force of Nature",
    "Spirit Visage",
    "Dead Man's Plate",
    "Unending Despair"
}

ADC_ITEMS = {
    "Infinity Edge",
    "Kraken Slayer",
    "The Collector",
    "Runaan's Hurricane",
    "Phantom Dancer",
    "Lord Dominik's Regards",
    "Mortal Reminder",
    "Bloodthirster",
    "Statikk Shiv",
    "Rapid Firecannon",
    "Guardian Angel",
    "Berserker's Greaves",
    "Blade of The Ruined King",
    "Immortal Shieldbow"
}

MAGE_ITEMS = {
    "Rabadon's Deathcap",
    "Luden's Companion",
    "Shadowflame",
    "Zhonya's Hourglass",
    "Void Staff",
    "Liandry's Torment",
    "Morellonomicon",
    "Banshee's Veil",
    "Stormsurge",
    "Malignance",
    "Blackfire Torch",
    "Rylai's Crystal Scepter"
}

BAD_RECOMMEND_ITEMS = {
    "Health Potion", "Refillable Potion", "Total Biscuit of Everlasting Will",
    "Stealth Ward", "Control Ward", "Oracle Lens", "Farsight Alteration",
    "Elixir of Iron", "Elixir of Sorcery", "Elixir of Wrath",

    "Doran's Shield", "Doran's Blade", "Doran's Ring",
    "Dark Seal", "Cull", "World Atlas",

    "Gustwalker Hatchling", "Mosstomper Seedling", "Scorchclaw Pup",

    "Boots", "Slightly Magical Footwear",

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
    "Cloak of Agility", "Steel Sigil"
}


def get_champion_role(champion):
    if champion in TANK_CHAMPS:
        return "tank"
    if champion in ADC_CHAMPS:
        return "adc"
    if champion in JUNGLE_MARKSMAN_CHAMPS:
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
        "ad_count": ad_count
    }


def is_valid_item_for_role(item_name, champion):
    if item_name in BAD_RECOMMEND_ITEMS:
        return False

    role = get_champion_role(champion)

    if role == "tank" and item_name not in TANK_ITEMS:
        return False

    if role == "adc" and item_name not in ADC_ITEMS:
        return False

    if role == "mage" and item_name not in MAGE_ITEMS:
        return False

    return True


def item_bonus(item_name, raw_row):
    champion = raw_row.get("champion", "")
    time_bucket = raw_row.get("time_bucket", "mid")
    threat = analyze_enemy_team(raw_row)
    role = get_champion_role(champion)

    bonus = 0.0

    # Boots are valid, but usually should not be the main recommendation late game.
    if item_name in COMPLETED_BOOTS:
        if time_bucket == "early":
            bonus += 0.15
        elif time_bucket == "mid":
            bonus -= 0.15
        elif time_bucket == "late":
            bonus -= 0.85

    # Scaling items are strongest when recommended early.
    if item_name in EARLY_SCALING_ITEMS:
        if time_bucket == "early":
            bonus += 0.35
        elif time_bucket == "mid":
            bonus -= 0.35
        elif time_bucket == "late":
            bonus -= 0.90

    # These are more useful as mid/late power recommendations.
    if item_name in LATE_GAME_POWER_ITEMS:
        if time_bucket == "mid":
            bonus += 0.15
        elif time_bucket == "late":
            bonus += 0.35

    if time_bucket in {"mid", "late"} and item_name in STARTER_ITEMS:
        bonus -= 0.50

    if threat["crit_count"] >= 2:
        if item_name == "Randuin's Omen":
            bonus += 1.00
        elif item_name == "Frozen Heart":
            bonus += 0.65
        elif item_name == "Thornmail":
            bonus += 0.35

    if threat["ad_count"] >= 3:
        if item_name in {
            "Randuin's Omen",
            "Frozen Heart",
            "Thornmail",
            "Sunfire Aegis"
        }:
            bonus += 0.25

    if threat["ap_count"] >= 3:
        if item_name in {
            "Kaenic Rookern",
            "Force of Nature",
            "Spirit Visage",
            "Hollow Radiance"
        }:
            bonus += 0.35

    if role == "tank" and item_name in TANK_ITEMS:
        bonus += 0.20

    if role == "adc" and item_name in ADC_ITEMS:
        bonus += 0.20

    if role == "mage" and item_name in MAGE_ITEMS:
        bonus += 0.20

    return bonus


def explain_item(item_name, raw_row):
    champion = raw_row.get("champion", "")
    lane = raw_row.get("lane", "")
    time_bucket = raw_row.get("time_bucket", "mid")
    threat = analyze_enemy_team(raw_row)
    role = get_champion_role(champion)

    enemies = [e for e in threat["enemies"] if e != "None"]

    intro = f"{item_name} is recommended for {champion}"

    context_parts = []

    if lane and lane != "NONE":
        context_parts.append(f"in the {lane} lane")

    if time_bucket == "early":
        context_parts.append("during the early game")
    elif time_bucket == "mid":
        context_parts.append("during the mid game")
    else:
        context_parts.append("during the late game")

    if context_parts:
        intro += " " + " ".join(context_parts)

    intro += "."

    reasons = []

    if role == "tank":
        reasons.append("As a tank, this recommendation prioritizes survivability and frontline value.")
    elif role == "adc":
        reasons.append("As a damage carry, this recommendation focuses on damage output while keeping you alive in fights.")
    elif role == "mage":
        reasons.append("As a mage, this recommendation supports stronger ability damage and safer teamfighting.")

    if threat["crit_count"] >= 2 and item_name == "Randuin's Omen":
        reasons.append("The enemy team has multiple critical-strike threats, so anti-crit armor is especially valuable.")

    elif threat["ad_count"] >= 3 and item_name in {
        "Frozen Heart", "Thornmail", "Sunfire Aegis", "Randuin's Omen"
    }:
        reasons.append("The enemy team is mostly physical damage, so armor is a high-value defensive stat.")

    elif threat["ap_count"] >= 3 and item_name in {
        "Kaenic Rookern", "Force of Nature", "Spirit Visage", "Hollow Radiance"
    }:
        reasons.append("The enemy team has heavy magic damage, so magic resistance is prioritized.")

    if role == "adc":
        if item_name == "Bloodthirster":
            reasons.append("Bloodthirster adds lifesteal and shielding, which helps survive against dive-heavy teams.")
        elif item_name == "Infinity Edge":
            reasons.append("Infinity Edge is a major damage spike for marksmen and improves carry potential.")
        elif item_name == "Lord Dominik's Regards":
            reasons.append("Lord Dominik's Regards helps cut through armor and stronger frontline targets.")
        elif item_name == "Kraken Slayer":
            reasons.append("Kraken Slayer is useful for sustained damage in longer fights.")
        elif item_name == "Guardian Angel":
            reasons.append("Guardian Angel is valuable when surviving one extra engage can decide a late-game fight.")

    if item_name in EARLY_SCALING_ITEMS and time_bucket in {"mid", "late"}:
        reasons.append("However, this item is usually strongest when purchased earlier, so it may be less ideal at this stage.")

    if item_name in COMPLETED_BOOTS and time_bucket == "late":
        reasons.append("Boots are useful, but late-game recommendations usually favor larger power items.")

    if not reasons:
        reasons.append("The neural network selected this item based on similar historical match states.")

    return " ".join([intro] + reasons[:3])