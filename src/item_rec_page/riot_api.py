from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests


APEX_ENDPOINTS = {
    "CHALLENGER": "challengerleagues",
    "GRANDMASTER": "grandmasterleagues",
    "MASTER": "masterleagues",
}

DD_VERSION_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DD_ITEM_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/item.json"


class RiotApiError(RuntimeError):
    pass


class RiotApiClient:
    def __init__(self, api_key: str, platform: str, region: str) -> None:
        self.api_key = api_key.strip()
        self.platform = platform.lower()
        self.region = region.lower()
        self.session = requests.Session()
        self.session.headers.update({"X-Riot-Token": self.api_key})

    @property
    def platform_base(self) -> str:
        return f"https://{self.platform}.api.riotgames.com"

    @property
    def regional_base(self) -> str:
        return f"https://{self.region}.api.riotgames.com"

    def _request_json(self, url: str, params: dict | None = None, use_auth: bool = True) -> dict | list:
        headers = self.session.headers if use_auth else None
        response = self.session.get(url, params=params, headers=headers, timeout=30)
        if response.status_code >= 400:
            if response.status_code == 401 and "apikey" in response.text.lower():
                raise RiotApiError(
                    "Riot API rejected the API key with 401 Unknown apikey. "
                    "Riot development keys expire every 24 hours, so generate a fresh key at "
                    "https://developer.riotgames.com/ and set it in the same shell before retrying."
                )
            raise RiotApiError(f"Riot API error {response.status_code} for {url}: {response.text[:300]}")
        return response.json()

    def get_league_entries(self, queue: str, tier: str, division: str) -> list[dict]:
        url = f"{self.platform_base}/lol/league/v4/entries/{queue}/{tier}/{division}"
        payload = self._request_json(url)
        return payload if isinstance(payload, list) else []

    def get_apex_league(self, queue: str, tier: str) -> list[dict]:
        endpoint = APEX_ENDPOINTS[tier]
        url = f"{self.platform_base}/lol/league/v4/{endpoint}/by-queue/{queue}"
        payload = self._request_json(url)
        entries = payload.get("entries", []) if isinstance(payload, dict) else []
        for entry in entries:
            entry.setdefault("tier", tier)
        return entries

    def iter_diamond_plus_players(self, queue: str, max_players: int) -> list[dict]:
        players: list[dict] = []
        seen_ids: set[str] = set()

        for tier in ("CHALLENGER", "GRANDMASTER", "MASTER"):
            for entry in self.get_apex_league(queue, tier):
                player_id = str(entry.get("puuid") or entry.get("summonerId") or "").strip()
                if player_id and player_id not in seen_ids:
                    seen_ids.add(player_id)
                    players.append(entry)
                    if len(players) >= max_players:
                        return players

        for division in ("I", "II", "III", "IV"):
            for entry in self.get_league_entries(queue, "DIAMOND", division):
                player_id = str(entry.get("puuid") or entry.get("summonerId") or "").strip()
                if player_id and player_id not in seen_ids:
                    seen_ids.add(player_id)
                    players.append(entry)
                    if len(players) >= max_players:
                        return players

        return players

    def get_summoner_by_id(self, encrypted_summoner_id: str) -> dict:
        encoded_id = quote(str(encrypted_summoner_id), safe="")
        url = f"{self.platform_base}/lol/summoner/v4/summoners/{encoded_id}"
        payload = self._request_json(url)
        return payload if isinstance(payload, dict) else {}

    def get_summoner_by_name(self, summoner_name: str) -> dict:
        encoded_name = quote(str(summoner_name), safe="")
        url = f"{self.platform_base}/lol/summoner/v4/summoners/by-name/{encoded_name}"
        payload = self._request_json(url)
        return payload if isinstance(payload, dict) else {}

    def get_match_ids_by_puuid(
        self,
        puuid: str,
        start: int = 0,
        count: int = 20,
        queue_id: int | None = 420,
    ) -> list[str]:
        url = f"{self.regional_base}/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"start": start, "count": count}
        if queue_id is not None:
            params["queue"] = queue_id
        payload = self._request_json(url, params=params)
        return payload if isinstance(payload, list) else []

    def get_match(self, match_id: str) -> dict:
        url = f"{self.regional_base}/lol/match/v5/matches/{match_id}"
        payload = self._request_json(url)
        return payload if isinstance(payload, dict) else {}

    def get_timeline(self, match_id: str) -> dict:
        url = f"{self.regional_base}/lol/match/v5/matches/{match_id}/timeline"
        payload = self._request_json(url)
        return payload if isinstance(payload, dict) else {}

    def get_latest_ddragon_version(self) -> str:
        payload = self._request_json(DD_VERSION_URL, use_auth=False)
        if not isinstance(payload, list) or not payload:
            raise RiotApiError("Could not resolve the latest Data Dragon version.")
        return str(payload[0])

    def get_item_catalog(self) -> dict[str, dict]:
        version = self.get_latest_ddragon_version()
        payload = self._request_json(DD_ITEM_URL.format(version=version), use_auth=False)
        if not isinstance(payload, dict):
            raise RiotApiError("Invalid item catalog payload from Data Dragon.")
        return {
            item_id: {
                **details,
                "id": item_id,
                "version": version,
            }
            for item_id, details in payload.get("data", {}).items()
        }


def save_item_catalog(catalog: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2)


def load_item_catalog(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def eligible_training_items(catalog: dict[str, dict] | None) -> set[str] | None:
    if not catalog:
        return None

    allowed: set[str] = set()
    for item_id, details in catalog.items():
        gold = details.get("gold", {})
        maps = details.get("maps", {})
        if not gold.get("purchasable", False):
            continue
        if details.get("consumed", False):
            continue
        if not maps.get("11", False):
            continue
        allowed.add(str(item_id))
    return allowed


def item_name_from_catalog(item_id: str | int, catalog: dict[str, dict] | None) -> str | None:
    if not catalog:
        return None
    details = catalog.get(str(item_id))
    return details.get("name") if details else None


def estimate_inventory_value(item_ids: Iterable[int], catalog: dict[str, dict] | None) -> float:
    if not catalog:
        return 0.0
    total = 0.0
    for item_id in item_ids:
        details = catalog.get(str(item_id))
        if details:
            total += float(details.get("gold", {}).get("total", 0.0))
    return total
