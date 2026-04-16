import requests

def get_item_names():
    try:
        versions = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()
        latest = versions[0]
        items = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{latest}/data/en_US/item.json").json()
        return {item_id: data["name"] for item_id, data in items["data"].items()}
    except:
        return {}

ITEM_NAMES = get_item_names()

def get_item_name(item_id):
    return ITEM_NAMES.get(str(item_id), f"Item {item_id}")