from __future__ import annotations

from typing import Any

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

from item_rec_page.config import Settings


requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


class LiveClientError(RuntimeError):
    pass


class LiveClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.session = requests.Session()

    def _get(self, endpoint: str) -> Any:
        url = f"{self.settings.liveclient_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        try:
            response = self.session.get(
                url,
                timeout=5,
                verify=self.settings.verify_liveclient_ssl,
            )
        except requests.RequestException as exc:
            raise LiveClientError(
                "Could not reach the local Live Client Data API. Make sure League is running and you are in an active game."
            ) from exc

        if response.status_code >= 400:
            raise LiveClientError(f"Live Client request failed for {endpoint}: {response.status_code} {response.text[:200]}")

        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            return response.json()
        text = response.text.strip()
        if text.startswith('"') and text.endswith('"'):
            return text.strip('"')
        return text

    def get_all_game_data(self) -> dict:
        payload = self._get("allgamedata")
        return payload if isinstance(payload, dict) else {}

    def get_active_player_name(self) -> str:
        payload = self._get("activeplayername")
        return str(payload)

    def get_live_snapshot(self) -> dict:
        all_game_data = self.get_all_game_data()
        active_player_name = self.get_active_player_name()
        game_data = all_game_data.get("gameData", {})
        players = all_game_data.get("allPlayers", [])
        active_player_details = all_game_data.get("activePlayer", {})

        summarized_players = [self._summarize_player(player) for player in players]
        active_player = next(
            (player for player in summarized_players if player["summoner_name"] == active_player_name),
            {"summoner_name": active_player_name, "item_ids": []},
        )
        active_player["current_gold"] = float(active_player_details.get("currentGold", 0.0))
        active_player["champion_stats"] = active_player_details.get("championStats", {})
        active_player["abilities"] = active_player_details.get("abilities", {})

        return {
            "game_mode": game_data.get("gameMode"),
            "map_number": game_data.get("mapNumber"),
            "game_time_seconds": float(game_data.get("gameTime", 0.0)),
            "active_player_name": active_player_name,
            "active_player": active_player,
            "players": summarized_players,
            "events": all_game_data.get("events", {}).get("Events", []),
        }

    @staticmethod
    def _summarize_player(player: dict) -> dict:
        scores = player.get("scores", {})
        return {
            "summoner_name": player.get("summonerName"),
            "champion_name": player.get("championName"),
            "team": player.get("team"),
            "level": int(player.get("level", 0)),
            "is_dead": bool(player.get("isDead", False)),
            "item_ids": [int(item.get("itemID", 0)) for item in player.get("items", []) if int(item.get("itemID", 0)) > 0],
            "items": [
                {
                    "item_id": int(item.get("itemID", 0)),
                    "display_name": item.get("displayName"),
                }
                for item in player.get("items", [])
                if int(item.get("itemID", 0)) > 0
            ],
            "scores": {
                "kills": int(scores.get("kills", 0)),
                "deaths": int(scores.get("deaths", 0)),
                "assists": int(scores.get("assists", 0)),
                "creep_score": int(scores.get("creepScore", 0)),
                "ward_score": int(scores.get("wardScore", 0)),
            },
        }
