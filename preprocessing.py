"""Preprocessing and parsing routines for blackjack, roulette, and poker."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from data_loading import iter_poker_games
from utils import flatten_nested_list, normalize_name, safe_literal_eval


BLACKJACK_ACTION_MAP = {
    "H": "hit",
    "S": "stand",
    "D": "double",
    "P": "split",
    "BJ": "blackjack",
}


def _blackjack_total(cards: list[int]) -> tuple[int, bool]:
    """Return blackjack total and whether the hand is soft."""
    total = sum(cards)
    aces = sum(card == 11 for card in cards)
    soft = aces > 0
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total, soft and total <= 21


def preprocess_blackjack(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Parse blackjack list-like fields into structured columns."""
    df = raw_df.copy()

    initial_cards = df["initial_hand"].apply(lambda value: safe_literal_eval(value, []))
    final_totals = df["player_final_value"].apply(lambda value: safe_literal_eval(value, []))
    action_lists = df["actions_taken"].apply(lambda value: safe_literal_eval(value, []))

    df["initial_total"] = initial_cards.apply(lambda cards: _blackjack_total(cards)[0] if cards else np.nan)
    df["soft_hand"] = initial_cards.apply(lambda cards: int(_blackjack_total(cards)[1]) if cards else 0)
    df["pair_hand"] = initial_cards.apply(
        lambda cards: int(len(cards) == 2 and len(set(cards)) == 1) if cards else 0
    )
    df["hand_strength"] = df["initial_total"] / 21.0
    df["dealer_risk"] = df["dealer_up"].map(lambda card: 1 if card in {2, 3, 4, 5, 6} else 0)
    df["natural_blackjack"] = initial_cards.apply(
        lambda cards: int(len(cards) == 2 and sorted(cards) == [10, 11]) if cards else 0
    )

    def parse_terminal_totals(value: Any) -> list[float]:
        parsed = flatten_nested_list(value if isinstance(value, list) else safe_literal_eval(value, []))
        totals: list[float] = []
        for element in parsed:
            if isinstance(element, (int, float)):
                totals.append(float(element))
            elif isinstance(element, str) and element == "BJ":
                totals.append(21.0)
        return totals

    df["terminal_totals"] = final_totals.apply(parse_terminal_totals)
    df["terminal_total_max"] = df["terminal_totals"].apply(lambda values: max(values) if values else np.nan)
    df["terminal_total_mean"] = df["terminal_totals"].apply(lambda values: float(np.mean(values)) if values else np.nan)

    flattened_actions = action_lists.apply(lambda actions: flatten_nested_list(actions))
    df["primary_action"] = flattened_actions.apply(
        lambda actions: BLACKJACK_ACTION_MAP.get(actions[0], "unknown") if actions else "unknown"
    )
    df["total_actions"] = flattened_actions.apply(len)
    for code, label in BLACKJACK_ACTION_MAP.items():
        df[f"num_{label}s"] = flattened_actions.apply(lambda actions, code=code: actions.count(code))

    df["outcome_label"] = np.select(
        [df["win"] > 0, df["win"] < 0],
        ["win", "loss"],
        default="draw",
    )
    df["target_win"] = (df["win"] > 0).astype(int)

    return df.drop(columns=["terminal_totals"])


def preprocess_roulette(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize roulette columns and derive canonical descriptors."""
    df = raw_df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    df["winning_color"] = df["winning_color"].str.lower()
    df["parity"] = np.where(df["winning_number"] == 0, "zero", np.where(df["winning_number"] % 2 == 0, "even", "odd"))
    df["range_bucket"] = np.where(
        df["winning_number"] == 0,
        "zero",
        np.where(df["winning_number"] <= 18, "low", "high"),
    )
    df["is_red"] = (df["winning_color"] == "red").astype(int)
    df["is_black"] = (df["winning_color"] == "black").astype(int)
    df["is_zero"] = (df["winning_number"] == 0).astype(int)
    return df


GAME_ID_RE = re.compile(
    r"Game ID:\s*(?P<game_id>\d+)\s+(?P<small_blind>\d+(?:\.\d+)?)\/(?P<big_blind>\d+(?:\.\d+)?)"
)
BUTTON_RE = re.compile(r"Seat\s+(?P<button>\d+)\s+is the button")
SEAT_RE = re.compile(r"Seat\s+(?P<seat>\d+):\s+(?P<player>.+?)\s+\((?P<stack>[\d.]+)\)\.")
BLIND_RE = re.compile(r"Player\s+(?P<player>.+?)\s+has\s+(?P<blind>small blind|big blind)\s+\((?P<amount>[\d.]+)\)")
ACTION_RE = re.compile(
    r"Player\s+(?P<player>.+?)\s+(?P<action>folds|calls|raises|checks|bets|allin|caps)(?:\s+\((?P<amount>[\d.]+)\))?"
)
UNCALLED_RE = re.compile(r"Uncalled bet\s+\((?P<amount>[\d.]+)\)\s+returned to\s+(?P<player>.+)")
SUMMARY_RE = re.compile(r"Bets:\s+(?P<bets>\d+(?:\.\d+)?)\.\s+Collects:\s+(?P<collects>\d+(?:\.\d+)?)\.")
POT_RE = re.compile(r"Pot:\s+(?P<pot>\d+(?:\.\d+)?)\.\s+Rake\s+(?P<rake>\d+(?:\.\d+)?)")


def parse_summary_player_line(line: str) -> dict[str, Any] | None:
    """Parse a poker summary line into player accounting fields."""
    if "Bets:" not in line or "Collects:" not in line or "Player " not in line:
        return None

    numeric_match = SUMMARY_RE.search(line)
    if not numeric_match:
        return None

    prefix = line.lstrip("*")
    if not prefix.startswith("Player "):
        return None

    player_part = prefix[len("Player ") :].split("Bets:", maxsplit=1)[0].strip()
    for marker in (" shows:", " mucks (does not show cards).", " mucks cards", " does not show cards."):
        if marker in player_part:
            player_part = player_part.split(marker, maxsplit=1)[0].strip()
            break

    return {
        "player": normalize_name(player_part.rstrip(".")),
        "bets": float(numeric_match.group("bets")),
        "collects": float(numeric_match.group("collects")),
        "shown": "shows:" in line,
        "mucked": "mucks" in line,
    }


def assign_positions(seat_map: dict[int, dict[str, Any]], button_seat: int) -> dict[str, str]:
    """Assign coarse poker positions from occupied seats and button seat."""
    seats = sorted(seat_map)
    if not seats:
        return {}

    button_idx = seats.index(button_seat) if button_seat in seat_map else 0
    order = seats[button_idx:] + seats[:button_idx]

    labels_by_offset: dict[int, str]
    if len(order) == 2:
        labels_by_offset = {0: "btn_sb", 1: "bb"}
    else:
        labels_by_offset = {0: "btn", 1: "sb", 2: "bb"}
        remaining = order[3:]
        if remaining:
            if len(remaining) == 1:
                labels_by_offset[3] = "utg"
            else:
                for idx, _ in enumerate(remaining, start=3):
                    if idx == len(order) - 1:
                        labels_by_offset[idx] = "co"
                    elif idx >= len(order) - 2:
                        labels_by_offset[idx] = "late"
                    elif idx <= 4:
                        labels_by_offset[idx] = "early"
                    else:
                        labels_by_offset[idx] = "middle"

    seat_to_label: dict[str, str] = {}
    for offset, seat in enumerate(order):
        player = seat_map[seat]["player"]
        seat_to_label[player] = labels_by_offset.get(offset, "middle")
    return seat_to_label


def parse_poker_logs(paths: list[str], max_games: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse unstructured poker hand histories into action-level and summary-level tables."""
    action_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

    for lines in iter_poker_games(paths, max_games=max_games):
        if len(lines) < 5:
            continue

        metadata: dict[str, Any] = {
            "game_id": None,
            "small_blind": np.nan,
            "big_blind": np.nan,
            "button_seat": None,
        }
        seat_map: dict[int, dict[str, Any]] = {}
        returned_uncalled: defaultdict[str, float] = defaultdict(float)
        game_action_records: list[dict[str, Any]] = []
        game_summary_records: list[dict[str, Any]] = []
        stage = "preflop"
        in_summary = False
        pot_amount = np.nan
        rake = np.nan

        for line in lines:
            if match := GAME_ID_RE.search(line):
                metadata["game_id"] = int(match.group("game_id"))
                metadata["small_blind"] = float(match.group("small_blind"))
                metadata["big_blind"] = float(match.group("big_blind"))
                continue

            if match := BUTTON_RE.search(line):
                metadata["button_seat"] = int(match.group("button"))
                continue

            if match := SEAT_RE.search(line):
                seat = int(match.group("seat"))
                player = normalize_name(match.group("player"))
                seat_map[seat] = {
                    "player": player,
                    "stack_size": float(match.group("stack")),
                }
                continue

            if line.startswith("------ Summary ------"):
                in_summary = True
                continue

            if line.startswith("*** FLOP ***"):
                stage = "flop"
                continue
            if line.startswith("*** TURN ***"):
                stage = "turn"
                continue
            if line.startswith("*** RIVER ***"):
                stage = "river"
                continue

            if in_summary and (match := POT_RE.search(line)):
                pot_amount = float(match.group("pot"))
                rake = float(match.group("rake"))
                continue

            if match := BLIND_RE.search(line):
                player = normalize_name(match.group("player"))
                game_action_records.append(
                    {
                        "game_id": metadata["game_id"],
                        "player": player,
                        "action": match.group("blind").replace(" ", "_"),
                        "bet_amount": float(match.group("amount")),
                        "stage": "blind",
                    }
                )
                continue

            if match := ACTION_RE.search(line):
                player = normalize_name(match.group("player"))
                game_action_records.append(
                    {
                        "game_id": metadata["game_id"],
                        "player": player,
                        "action": match.group("action"),
                        "bet_amount": float(match.group("amount")) if match.group("amount") else 0.0,
                        "stage": stage,
                    }
                )
                continue

            if match := UNCALLED_RE.search(line):
                returned_uncalled[normalize_name(match.group("player"))] += float(match.group("amount"))
                continue

            parsed_summary = parse_summary_player_line(line) if in_summary else None
            if parsed_summary:
                player = parsed_summary["player"]
                bets = parsed_summary["bets"]
                collects = parsed_summary["collects"]
                profit = collects - bets
                shown = parsed_summary["shown"]
                mucked = parsed_summary["mucked"]
                game_summary_records.append(
                    {
                        "game_id": metadata["game_id"],
                        "player": player,
                        "bets": bets,
                        "collects": collects,
                        "profit": profit,
                        "returned_uncalled": returned_uncalled[player],
                        "went_to_showdown": int(shown or mucked),
                        "showed_cards": int(shown),
                        "mucked": int(mucked),
                        "pot_amount": pot_amount,
                        "rake": rake,
                        "small_blind": metadata["small_blind"],
                        "big_blind": metadata["big_blind"],
                        "button_seat": metadata["button_seat"],
                    }
                )

        if not metadata["game_id"] or not seat_map:
            continue

        seat_to_position = assign_positions(seat_map, int(metadata["button_seat"]) if metadata["button_seat"] else 0)
        player_lookup = {
            details["player"]: {
                "seat": seat,
                "stack_size": details["stack_size"],
                "position": seat_to_position.get(details["player"], "unknown"),
            }
            for seat, details in seat_map.items()
        }

        for record in game_action_records:
            player_info = player_lookup.get(record["player"], {"seat": np.nan, "stack_size": np.nan, "position": "unknown"})
            record["seat"] = player_info["seat"]
            record["stack_size"] = player_info["stack_size"]
            record["position"] = player_info["position"]
            record["small_blind"] = metadata["small_blind"]
            record["big_blind"] = metadata["big_blind"]
        action_records.extend(game_action_records)

        for record in game_summary_records:
            player_info = player_lookup.get(record["player"], {"seat": np.nan, "stack_size": np.nan, "position": "unknown"})
            record["seat"] = player_info["seat"]
            record["stack_size"] = player_info["stack_size"]
            record["position"] = player_info["position"]
        summary_records.extend(game_summary_records)

    action_df = pd.DataFrame(action_records)
    summary_df = pd.DataFrame(summary_records)

    if action_df.empty or summary_df.empty:
        raise ValueError("Poker parsing did not produce any structured records.")

    for column in ("bet_amount", "stack_size", "small_blind", "big_blind"):
        if column in action_df:
            action_df[column] = pd.to_numeric(action_df[column], errors="coerce").fillna(0.0)

    for column in ("bets", "collects", "profit", "returned_uncalled", "stack_size", "pot_amount", "rake", "small_blind", "big_blind"):
        if column in summary_df:
            summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce").fillna(0.0)

    return action_df, summary_df
