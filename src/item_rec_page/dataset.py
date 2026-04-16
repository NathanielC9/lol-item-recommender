from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from item_rec_page.riot_api import RiotApiClient, eligible_training_items


ROLE_ALIASES = {
    "TOP": "TOP",
    "JUNGLE": "JUNGLE",
    "MIDDLE": "MIDDLE",
    "MID": "MIDDLE",
    "BOTTOM": "BOTTOM",
    "ADC": "BOTTOM",
    "UTILITY": "UTILITY",
    "SUPPORT": "UTILITY",
}

INVENTORY_COLUMNS = [f"inventory_{index}" for index in range(6)]
DATASET_COLUMNS = [
    "match_id",
    "participant_id",
    "champion",
    "role",
    "game_time_seconds",
    "level",
    "current_gold",
    "total_gold",
    "cs",
    "jungle_cs",
    "gold_diff_vs_role",
    "level_diff_vs_role",
    "cs_diff_vs_role",
    "team_gold_diff",
    "team_level_diff",
    *INVENTORY_COLUMNS,
    "label_item_id",
]


@dataclass(frozen=True)
class CollectionConfig:
    queue: str
    queue_id: int
    max_seed_players: int
    matches_per_player: int
    max_matches: int
    output_path: Path
    item_catalog: dict[str, dict] | None = None


@dataclass
class MatchExtractionStats:
    missing_participants: bool = False
    missing_frames: bool = False
    purchase_events_seen: int = 0
    allowed_purchase_events: int = 0
    purchase_events_missing_participant: int = 0
    rows_skipped_missing_participant_frame: int = 0
    rows_built: int = 0


@dataclass
class CollectionStats:
    seed_players_seen: int = 0
    entry_puuid_available: int = 0
    summoner_lookup_missing_puuid: int = 0
    summoner_name_fallback_attempts: int = 0
    summoner_name_fallback_successes: int = 0
    players_with_puuid: int = 0
    players_with_ranked_matches: int = 0
    candidate_match_ids_seen: int = 0
    unique_matches_seen: int = 0
    queue_matched_matches: int = 0
    matches_skipped_wrong_queue: int = 0
    matches_missing_participants: int = 0
    matches_missing_frames: int = 0
    matches_with_rows: int = 0
    item_purchase_events_seen: int = 0
    allowed_item_purchase_events: int = 0
    purchase_events_missing_participant: int = 0
    rows_skipped_missing_participant_frame: int = 0
    rows_written: int = 0


@dataclass(frozen=True)
class CollectionResult:
    dataframe: pd.DataFrame
    stats: CollectionStats


def normalize_role(raw_role: str | None) -> str:
    if not raw_role:
        return "UNKNOWN"
    normalized = str(raw_role).strip().upper()
    return ROLE_ALIASES.get(normalized, "UNKNOWN")


def collect_training_examples(client: RiotApiClient, config: CollectionConfig) -> CollectionResult:
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    allowed_items = eligible_training_items(config.item_catalog)

    seen_match_ids: set[str] = set()
    collected_rows: list[dict] = []
    stats = CollectionStats()

    for entry in client.iter_diamond_plus_players(config.queue, config.max_seed_players):
        stats.seed_players_seen += 1
        puuid = resolve_entry_puuid(client, entry, stats)
        if not puuid:
            continue
        stats.players_with_puuid += 1

        match_ids = get_candidate_match_ids(client, puuid, config)
        stats.candidate_match_ids_seen += len(match_ids)
        if match_ids:
            stats.players_with_ranked_matches += 1

        queue_matches_collected = 0
        for match_id in match_ids:
            if match_id in seen_match_ids:
                continue

            match = client.get_match(match_id)
            if int(match.get("info", {}).get("queueId", -1)) != config.queue_id:
                stats.matches_skipped_wrong_queue += 1
                continue

            queue_matches_collected += 1
            seen_match_ids.add(match_id)
            stats.unique_matches_seen += 1
            stats.queue_matched_matches += 1

            timeline = client.get_timeline(match_id)
            match_rows, match_stats = extract_examples_from_match(match, timeline, allowed_items)
            collected_rows.extend(match_rows)
            if match_stats.missing_participants:
                stats.matches_missing_participants += 1
            if match_stats.missing_frames:
                stats.matches_missing_frames += 1
            stats.item_purchase_events_seen += match_stats.purchase_events_seen
            stats.allowed_item_purchase_events += match_stats.allowed_purchase_events
            stats.purchase_events_missing_participant += match_stats.purchase_events_missing_participant
            stats.rows_skipped_missing_participant_frame += match_stats.rows_skipped_missing_participant_frame
            if match_stats.rows_built > 0:
                stats.matches_with_rows += 1

            if len(seen_match_ids) >= config.max_matches:
                return finalize_collection(collected_rows, config.output_path, stats)

            if queue_matches_collected >= config.matches_per_player:
                break

    return finalize_collection(collected_rows, config.output_path, stats)


def resolve_entry_puuid(client: RiotApiClient, entry: dict, stats: CollectionStats) -> str | None:
    entry_puuid = entry.get("puuid")
    if entry_puuid:
        stats.entry_puuid_available += 1
        return str(entry_puuid)

    summoner_id = entry.get("summonerId")
    summoner_name = entry.get("summonerName")

    summoner: dict = {}
    if summoner_id:
        summoner = client.get_summoner_by_id(str(summoner_id))
        puuid = summoner.get("puuid")
        if puuid:
            return str(puuid)
        stats.summoner_lookup_missing_puuid += 1

    if summoner_name:
        stats.summoner_name_fallback_attempts += 1
        summoner = client.get_summoner_by_name(str(summoner_name))
        puuid = summoner.get("puuid")
        if puuid:
            stats.summoner_name_fallback_successes += 1
            return str(puuid)

    return None


def get_candidate_match_ids(client: RiotApiClient, puuid: str, config: CollectionConfig) -> list[str]:
    primary_match_ids = client.get_match_ids_by_puuid(
        puuid,
        start=0,
        count=config.matches_per_player,
        queue_id=config.queue_id,
    )
    if len(primary_match_ids) >= config.matches_per_player:
        return primary_match_ids

    fallback_count = min(max(config.matches_per_player * 5, 20), 100)
    fallback_match_ids = client.get_match_ids_by_puuid(
        puuid,
        start=0,
        count=fallback_count,
        queue_id=None,
    )

    merged: list[str] = []
    seen_ids: set[str] = set()
    for match_id in primary_match_ids + fallback_match_ids:
        if match_id in seen_ids:
            continue
        seen_ids.add(match_id)
        merged.append(match_id)
    return merged


def finalize_collection(rows: list[dict], output_path: Path, stats: CollectionStats) -> CollectionResult:
    dataframe = pd.DataFrame(rows, columns=DATASET_COLUMNS)
    stats.rows_written = int(len(dataframe))
    dataframe.to_csv(output_path, index=False)
    return CollectionResult(dataframe=dataframe, stats=stats)


def extract_examples_from_match(
    match_payload: dict,
    timeline_payload: dict,
    allowed_items: set[str] | None = None,
) -> tuple[list[dict], MatchExtractionStats]:
    info = match_payload.get("info", {})
    timeline_info = timeline_payload.get("info", {})
    participants = info.get("participants", [])
    frames = timeline_info.get("frames", [])

    if not participants or not frames:
        return [], MatchExtractionStats(
            missing_participants=not bool(participants),
            missing_frames=not bool(frames),
        )

    match_id = match_payload.get("metadata", {}).get("matchId")
    participant_map = {int(participant["participantId"]): participant for participant in participants}
    team_lookup = {
        int(participant["participantId"]): int(participant["teamId"])
        for participant in participants
    }
    opponent_map = build_role_opponent_map(participants)
    frame_timestamps = [int(frame.get("timestamp", 0)) for frame in frames]
    inventories = {participant_id: [] for participant_id in participant_map}

    rows: list[dict] = []
    stats = MatchExtractionStats()
    all_events = []
    for frame in frames:
        all_events.extend(frame.get("events", []))
    all_events.sort(key=lambda event: int(event.get("timestamp", 0)))

    for event in all_events:
        raw_participant_id = event.get("participantId")
        try:
            participant_id = int(raw_participant_id)
        except (TypeError, ValueError):
            if event.get("type") == "ITEM_PURCHASED":
                stats.purchase_events_missing_participant += 1
            continue
        if participant_id not in participant_map:
            if event.get("type") == "ITEM_PURCHASED":
                stats.purchase_events_missing_participant += 1
            continue

        event_type = event.get("type")
        if event_type == "ITEM_PURCHASED":
            stats.purchase_events_seen += 1
            item_id = int(event.get("itemId", 0))
            if item_id <= 0:
                continue

            if allowed_items is None or str(item_id) in allowed_items:
                stats.allowed_purchase_events += 1
                frame = find_frame_for_timestamp(frames, frame_timestamps, int(event.get("timestamp", 0)))
                row, missing_participant_frame = build_training_row(
                    match_id=match_id,
                    event_timestamp=int(event.get("timestamp", 0)),
                    participant=participant_map[participant_id],
                    frame=frame,
                    opponent_id=opponent_map.get(participant_id),
                    team_lookup=team_lookup,
                    inventory=inventories[participant_id],
                )
                if missing_participant_frame:
                    stats.rows_skipped_missing_participant_frame += 1
                if row is not None:
                    row["label_item_id"] = str(item_id)
                    rows.append(row)
                    stats.rows_built += 1

            add_item(inventories[participant_id], item_id)
        elif event_type in {"ITEM_DESTROYED", "ITEM_SOLD"}:
            remove_item(inventories[participant_id], int(event.get("itemId", 0)))
        elif event_type == "ITEM_UNDO":
            after_id = int(event.get("afterId", 0))
            before_id = int(event.get("beforeId", 0))
            if after_id > 0:
                remove_item(inventories[participant_id], after_id)
            if before_id > 0:
                add_item(inventories[participant_id], before_id)

    return rows, stats


def build_role_opponent_map(participants: list[dict]) -> dict[int, int]:
    by_team_and_role: dict[tuple[int, str], int] = {}
    for participant in participants:
        role = normalize_role(participant.get("teamPosition") or participant.get("individualPosition"))
        if role == "UNKNOWN":
            continue
        by_team_and_role[(int(participant["teamId"]), role)] = int(participant["participantId"])

    opponent_map: dict[int, int] = {}
    for participant in participants:
        participant_id = int(participant["participantId"])
        team_id = int(participant["teamId"])
        role = normalize_role(participant.get("teamPosition") or participant.get("individualPosition"))
        if role == "UNKNOWN":
            continue
        opposing_team_id = 200 if team_id == 100 else 100
        opponent_id = by_team_and_role.get((opposing_team_id, role))
        if opponent_id is not None:
            opponent_map[participant_id] = opponent_id
    return opponent_map


def find_frame_for_timestamp(frames: list[dict], frame_timestamps: list[int], timestamp: int) -> dict:
    index = bisect_right(frame_timestamps, timestamp) - 1
    if index < 0:
        return frames[0]
    return frames[index]


def build_training_row(
    match_id: str,
    event_timestamp: int,
    participant: dict,
    frame: dict,
    opponent_id: int | None,
    team_lookup: dict[int, int],
    inventory: list[int],
) -> tuple[dict | None, bool]:
    participant_frames = frame.get("participantFrames", {})
    participant_frame = participant_frames.get(str(participant["participantId"])) or participant_frames.get(participant["participantId"])
    if not participant_frame:
        return None, True

    team_id = int(participant["teamId"])
    participant_id = int(participant["participantId"])

    opponent_frame = None
    if opponent_id is not None:
        opponent_frame = participant_frames.get(str(opponent_id)) or participant_frames.get(opponent_id)

    team_gold, team_level = aggregate_team_state(participant_frames, team_lookup, team_id)
    enemy_gold, enemy_level = aggregate_team_state(participant_frames, team_lookup, 200 if team_id == 100 else 100)

    minions_killed = int(participant_frame.get("minionsKilled", 0))
    jungle_kills = int(participant_frame.get("jungleMinionsKilled", 0))
    cs_total = minions_killed + jungle_kills

    row = {
        "match_id": match_id,
        "participant_id": participant_id,
        "champion": participant.get("championName"),
        "role": normalize_role(participant.get("teamPosition") or participant.get("individualPosition")),
        "game_time_seconds": event_timestamp / 1000.0,
        "level": float(participant_frame.get("level", 0)),
        "current_gold": float(participant_frame.get("currentGold", 0.0)),
        "total_gold": float(participant_frame.get("totalGold", 0.0)),
        "cs": float(cs_total),
        "jungle_cs": float(jungle_kills),
        "gold_diff_vs_role": 0.0,
        "level_diff_vs_role": 0.0,
        "cs_diff_vs_role": 0.0,
        "team_gold_diff": float(team_gold - enemy_gold),
        "team_level_diff": float(team_level - enemy_level),
    }

    if opponent_frame:
        opponent_cs = int(opponent_frame.get("minionsKilled", 0)) + int(opponent_frame.get("jungleMinionsKilled", 0))
        row["gold_diff_vs_role"] = float(participant_frame.get("totalGold", 0.0) - opponent_frame.get("totalGold", 0.0))
        row["level_diff_vs_role"] = float(participant_frame.get("level", 0.0) - opponent_frame.get("level", 0.0))
        row["cs_diff_vs_role"] = float(cs_total - opponent_cs)

    inventory_snapshot = padded_inventory(inventory)
    for column, value in zip(INVENTORY_COLUMNS, inventory_snapshot):
        row[column] = str(value)

    return row, False


def aggregate_team_state(participant_frames: dict, team_lookup: dict[int, int], team_id: int) -> tuple[float, float]:
    team_gold = 0.0
    team_level = 0.0
    for participant_id, raw_frame in participant_frames.items():
        participant_data = raw_frame or {}
        resolved_participant_id = int(participant_id)
        if team_lookup.get(resolved_participant_id) == team_id:
            team_gold += float(participant_data.get("totalGold", 0.0))
            team_level += float(participant_data.get("level", 0.0))
    return team_gold, team_level


def add_item(inventory: list[int], item_id: int) -> None:
    if item_id > 0:
        inventory.append(item_id)


def remove_item(inventory: list[int], item_id: int) -> None:
    try:
        inventory.remove(item_id)
    except ValueError:
        return


def padded_inventory(inventory: list[int]) -> list[int]:
    snapshot = inventory[:6]
    while len(snapshot) < 6:
        snapshot.append(0)
    return snapshot
