import requests

DDRAGON_BASE = "https://ddragon.leagueoflegends.com"


def _load_ddragon_data():
    try:
        versions = requests.get(f"{DDRAGON_BASE}/api/versions.json", timeout=5).json()
        latest = versions[0]
        item_data = requests.get(
            f"{DDRAGON_BASE}/cdn/{latest}/data/en_US/item.json",
            timeout=5
        ).json()["data"]
        champion_data = requests.get(
            f"{DDRAGON_BASE}/cdn/{latest}/data/en_US/champion.json",
            timeout=5
        ).json()["data"]
        return latest, item_data, champion_data
    except Exception:
        return "15.1.1", {}, {}


DDRAGON_VERSION, ITEM_DATA, CHAMPION_DATA = _load_ddragon_data()
ITEM_NAMES = {item_id: data.get("name", f"Item {item_id}") for item_id, data in ITEM_DATA.items()}
ITEM_NAME_TO_ID = {name: item_id for item_id, name in ITEM_NAMES.items()}
CHAMPION_NAME_TO_ID = {
    data.get("name", champ_id): champ_id
    for champ_id, data in CHAMPION_DATA.items()
}

# Common names in Riot data that differ from simple display text.
CHAMPION_ALIASES = {
    "Aurelion Sol": "AurelionSol",
    "Bel'Veth": "Belveth",
    "Cho'Gath": "Chogath",
    "Dr. Mundo": "DrMundo",
    "Jarvan IV": "JarvanIV",
    "Kai'Sa": "Kaisa",
    "Kha'Zix": "Khazix",
    "Kog'Maw": "KogMaw",
    "K'Sante": "KSante",
    "LeBlanc": "Leblanc",
    "Lee Sin": "LeeSin",
    "Master Yi": "MasterYi",
    "Miss Fortune": "MissFortune",
    "Nunu & Willump": "Nunu",
    "Rek'Sai": "RekSai",
    "Renata Glasc": "Renata",
    "Tahm Kench": "TahmKench",
    "Twisted Fate": "TwistedFate",
    "Vel'Koz": "Velkoz",
    "Wukong": "MonkeyKing",
    "Xin Zhao": "XinZhao",
}


def get_data_dragon_version():
    return DDRAGON_VERSION


def get_item_name(item_id):
    item_id = str(item_id)

    # If model already returns item name, return it directly.
    if not item_id.isdigit():
        return item_id

    return ITEM_NAMES.get(item_id, f"Item {item_id}")


def get_item_icon_url(item_id):
    item_id = str(item_id)

    if item_id.isdigit():
        return f"{DDRAGON_BASE}/cdn/{DDRAGON_VERSION}/img/item/{item_id}.png"

    riot_id = ITEM_NAME_TO_ID.get(item_id)

    if riot_id:
        return f"{DDRAGON_BASE}/cdn/{DDRAGON_VERSION}/img/item/{riot_id}.png"

    return ""

def get_item_details(item_id):
    item_id = str(item_id)

    # Convert name → ID if needed
    if not item_id.isdigit():
        item_id = ITEM_NAME_TO_ID.get(item_id, item_id)

    data = ITEM_DATA.get(item_id, {})

    gold = data.get("gold", {})
    maps = data.get("maps", {})

    return {
        "description": data.get("plaintext") or "No item description available.",
        "cost": gold.get("total"),
        "sell": gold.get("sell"),
        "purchasable": bool(gold.get("purchasable", False)),
        "tags": data.get("tags", []),
        "map11": bool(maps.get("11", True)),
    }


def _normalize_champion_id(champion):
    champion = str(champion or "").strip()

    if champion in CHAMPION_DATA:
        return champion
    if champion in CHAMPION_NAME_TO_ID:
        return CHAMPION_NAME_TO_ID[champion]
    if champion in CHAMPION_ALIASES:
        return CHAMPION_ALIASES[champion]

    compact = "".join(ch for ch in champion if ch.isalnum())
    if compact in CHAMPION_DATA:
        return compact

    return compact


def get_champion_icon_url(champion):
    champ_id = _normalize_champion_id(champion)
    if not champ_id:
        return ""
    return f"{DDRAGON_BASE}/cdn/{DDRAGON_VERSION}/img/champion/{champ_id}.png"
