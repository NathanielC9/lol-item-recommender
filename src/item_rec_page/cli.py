from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from item_rec_page.config import Settings
from item_rec_page.dataset import CollectionConfig, CollectionStats, collect_training_examples
from item_rec_page.live_client import LiveClient, LiveClientError
from item_rec_page.riot_api import RiotApiClient, RiotApiError, save_item_catalog


def build_parser() -> argparse.ArgumentParser:
    settings = Settings.from_env()

    parser = argparse.ArgumentParser(description="League item recommendation research CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    live_parser = subparsers.add_parser(
        "live",
        help="Monitor the Riot Live Client API and resume automatically when a game becomes available.",
    )
    live_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON snapshots while a game is active.")
    live_parser.add_argument("--once", action="store_true", help="Fetch a single snapshot and exit.")
    live_parser.add_argument("--poll-interval", type=float, default=3.0, help="Seconds between availability checks.")

    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect diamond+ ranked match timelines from Riot's official APIs and build training rows.",
    )
    collect_parser.add_argument("--api-key", default=settings.riot_api_key)
    collect_parser.add_argument("--platform", default=settings.platform)
    collect_parser.add_argument("--region", default=settings.region)
    collect_parser.add_argument("--queue", default="RANKED_SOLO_5x5")
    collect_parser.add_argument("--queue-id", type=int, default=420)
    collect_parser.add_argument("--max-players", type=int, default=120)
    collect_parser.add_argument("--matches-per-player", type=int, default=15)
    collect_parser.add_argument("--max-matches", type=int, default=500)
    collect_parser.add_argument(
        "--output",
        default=str(settings.data_dir / "training_examples.csv"),
    )

    train_parser = subparsers.add_parser("train", help="Train the Keras model from collected examples.")
    train_parser.add_argument(
        "--dataset",
        default=str(settings.data_dir / "training_examples.csv"),
    )
    train_parser.add_argument("--model-dir", default=str(settings.model_dir))
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=256)

    predict_parser = subparsers.add_parser(
        "predict-live",
        help="Predict likely next purchases from the active live-client game.",
    )
    predict_parser.add_argument("--role", default="UNKNOWN")
    predict_parser.add_argument("--model-dir", default=str(settings.model_dir))
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for an active game instead of exiting immediately if the Live Client API is unavailable.",
    )
    predict_parser.add_argument("--poll-interval", type=float, default=3.0)

    return parser


def print_json(payload: dict, pretty: bool = True) -> None:
    if pretty:
        print(json.dumps(payload, indent=2, sort_keys=False))
    else:
        print(json.dumps(payload))


def diagnose_collection(stats: CollectionStats, rows_written: int) -> str | None:
    if rows_written > 0:
        return None
    if stats.seed_players_seen == 0:
        return "The seeding endpoints returned no usable players for the requested queue."
    if stats.summoner_name_fallback_successes > 0 and stats.players_with_puuid > 0:
        return "Some PUUIDs were recovered through Riot's deprecated summoner-name lookup, but later stages still produced no rows."
    if stats.players_with_puuid == 0:
        if stats.summoner_name_fallback_attempts > 0:
            return "Seed players were fetched, but neither summoner-ID lookup nor the summoner-name fallback returned a PUUID."
        return "Seed players were fetched, but none resolved to a PUUID. Riot's summoner lookup may have changed or returned incomplete data."
    if stats.candidate_match_ids_seen == 0:
        return "Sampled players resolved correctly, but Riot returned no recent matches for them."
    if stats.queue_matched_matches == 0 and stats.matches_skipped_wrong_queue > 0:
        return "Recent matches were found, but none matched the requested queue ID. Try a larger sample or confirm the queue settings."
    if stats.matches_missing_frames > 0 and stats.item_purchase_events_seen == 0:
        return "Queue-matched matches were fetched, but the timelines had no frames. Riot may be returning incomplete timeline payloads for these matches."
    if stats.item_purchase_events_seen == 0 and stats.queue_matched_matches > 0:
        return "Queue-matched timelines were fetched, but no ITEM_PURCHASED events were found."
    if stats.allowed_item_purchase_events == 0 and stats.item_purchase_events_seen > 0:
        return "Purchase events were found, but every purchased item was filtered out by the allowed-item rules."
    if stats.rows_skipped_missing_participant_frame > 0 and stats.allowed_item_purchase_events > 0:
        return "Allowed purchase events were found, but participant frames were missing at the purchase timestamps."
    if stats.purchase_events_missing_participant > 0 and stats.item_purchase_events_seen == 0:
        return "Purchase events were present, but their participant IDs did not line up with the participant map."
    return "The collector reached queue-matched matches but still produced no rows. The new counters should narrow down which stage is dropping data."


def _snapshot_summary(snapshot: dict) -> str:
    active = snapshot["active_player"]
    scores = active.get("scores", {})
    return (
        f"[live] {active.get('summoner_name')} on {active.get('champion_name')} "
        f"| lvl {active.get('level', 0)} | K/D/A {scores.get('kills', 0)}/{scores.get('deaths', 0)}/{scores.get('assists', 0)} "
        f"| CS {scores.get('creep_score', 0)} | gold {round(float(active.get('current_gold', 0.0)), 1)} "
        f"| game_time {round(float(snapshot.get('game_time_seconds', 0.0)), 1)}s"
    )


def wait_for_live_snapshot(client: LiveClient, poll_interval: float) -> dict:
    unavailable_reported = False
    while True:
        try:
            snapshot = client.get_live_snapshot()
            if unavailable_reported:
                print("Live Client detected. Resuming.")
            return snapshot
        except LiveClientError:
            if not unavailable_reported:
                print("Live Client not detected. Waiting for an active League game...")
                unavailable_reported = True
            time.sleep(poll_interval)


def run_live_command(args: argparse.Namespace) -> None:
    client = LiveClient()

    if args.once:
        try:
            snapshot = client.get_live_snapshot()
        except LiveClientError as exc:
            print(str(exc))
            return
        print_json(snapshot, pretty=args.pretty)
        return

    print("Monitoring Riot Live Client API. Press Ctrl+C to stop.")
    availability_state: bool | None = None

    try:
        while True:
            try:
                snapshot = client.get_live_snapshot()
                if availability_state is not True:
                    print("Live Client detected.")
                availability_state = True
                if args.pretty:
                    print_json(snapshot, pretty=True)
                else:
                    print(_snapshot_summary(snapshot))
            except LiveClientError:
                if availability_state is True:
                    print("Live Client unavailable. Waiting for the next active game...")
                elif availability_state is None:
                    print("Live Client not detected yet. Waiting...")
                availability_state = False
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("Stopped live monitoring.")


def run_collect_command(args: argparse.Namespace) -> None:
    api_key = args.api_key.strip() if isinstance(args.api_key, str) else args.api_key
    if not api_key:
        raise SystemExit("Missing Riot API key. Set RIOT_API_KEY or pass --api-key.")

    settings = Settings.from_env(
        riot_api_key=api_key,
        platform=args.platform,
        region=args.region,
    )
    settings.ensure_directories()

    client = RiotApiClient(
        api_key=settings.riot_api_key,
        platform=settings.platform,
        region=settings.region,
    )
    try:
        item_catalog = client.get_item_catalog()
        save_item_catalog(item_catalog, settings.item_catalog_path)

        config = CollectionConfig(
            queue=args.queue,
            queue_id=args.queue_id,
            max_seed_players=args.max_players,
            matches_per_player=args.matches_per_player,
            max_matches=args.max_matches,
            output_path=Path(args.output),
            item_catalog=item_catalog,
        )
        result = collect_training_examples(client, config)
    except RiotApiError as exc:
        raise SystemExit(str(exc)) from exc

    dataframe = result.dataframe
    stats = result.stats
    print_json(
        {
            "rows": int(len(dataframe)),
            "output": str(config.output_path),
            "unique_labels": int(dataframe["label_item_id"].nunique()) if not dataframe.empty else 0,
            "unique_champions": int(dataframe["champion"].nunique()) if not dataframe.empty else 0,
            "seed_players_seen": stats.seed_players_seen,
            "entry_puuid_available": stats.entry_puuid_available,
            "summoner_lookup_missing_puuid": stats.summoner_lookup_missing_puuid,
            "summoner_name_fallback_attempts": stats.summoner_name_fallback_attempts,
            "summoner_name_fallback_successes": stats.summoner_name_fallback_successes,
            "players_with_puuid": stats.players_with_puuid,
            "players_with_ranked_matches": stats.players_with_ranked_matches,
            "candidate_match_ids_seen": stats.candidate_match_ids_seen,
            "unique_matches_seen": stats.unique_matches_seen,
            "queue_matched_matches": stats.queue_matched_matches,
            "matches_skipped_wrong_queue": stats.matches_skipped_wrong_queue,
            "matches_missing_participants": stats.matches_missing_participants,
            "matches_missing_frames": stats.matches_missing_frames,
            "matches_with_rows": stats.matches_with_rows,
            "item_purchase_events_seen": stats.item_purchase_events_seen,
            "allowed_item_purchase_events": stats.allowed_item_purchase_events,
            "purchase_events_missing_participant": stats.purchase_events_missing_participant,
            "rows_skipped_missing_participant_frame": stats.rows_skipped_missing_participant_frame,
            "diagnosis": diagnose_collection(stats, int(len(dataframe))),
        }
    )


def run_train_command(args: argparse.Namespace) -> None:
    from item_rec_page.modeling import train_model

    metrics = train_model(
        dataset_path=Path(args.dataset),
        model_dir=Path(args.model_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print_json(metrics)


def run_predict_live_command(args: argparse.Namespace) -> None:
    from item_rec_page.modeling import predict_live_snapshot

    client = LiveClient()
    try:
        if args.wait:
            snapshot = wait_for_live_snapshot(client, args.poll_interval)
        else:
            snapshot = client.get_live_snapshot()
    except KeyboardInterrupt:
        print("Stopped waiting for Live Client.")
        return
    except LiveClientError as exc:
        print(str(exc))
        return

    predictions = predict_live_snapshot(
        snapshot=snapshot,
        role=args.role,
        model_dir=Path(args.model_dir),
        top_k=args.top_k,
    )
    print_json(predictions)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "live":
        run_live_command(args)
    elif args.command == "collect":
        run_collect_command(args)
    elif args.command == "train":
        run_train_command(args)
    elif args.command == "predict-live":
        run_predict_live_command(args)
