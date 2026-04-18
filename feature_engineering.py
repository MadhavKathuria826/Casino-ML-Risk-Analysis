"""Feature engineering routines for the casino ML project."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import shannon_entropy


def engineer_blackjack_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select model-ready blackjack features and target."""
    feature_df = df[
        [
            "cards_remaining",
            "dealer_up",
            "run_count",
            "true_count",
            "initial_total",
            "soft_hand",
            "pair_hand",
            "hand_strength",
            "dealer_risk",
            "natural_blackjack",
            "terminal_total_max",
            "terminal_total_mean",
            "primary_action",
            "total_actions",
            "num_hits",
            "num_stands",
            "num_doubles",
            "num_splits",
        ]
    ].copy()
    feature_df = feature_df.fillna(0)
    target = df["target_win"].astype(int)
    return feature_df, target


def engineer_roulette_features(df: pd.DataFrame, window: int = 8) -> tuple[pd.DataFrame, pd.Series]:
    """Build sequence-derived roulette features to test predictability."""
    records: list[dict[str, Any]] = []

    numbers = df["winning_number"].tolist()
    colors = df["winning_color"].tolist()
    parities = df["parity"].tolist()
    ranges = df["range_bucket"].tolist()

    for idx in range(window, len(df) - 1):
        prev_numbers = numbers[idx - window : idx]
        prev_colors = colors[idx - window : idx]
        prev_parities = parities[idx - window : idx]
        prev_ranges = ranges[idx - window : idx]

        color_streak = 1
        for j in range(idx - 1, idx - window, -1):
            if colors[j] == colors[j - 1]:
                color_streak += 1
            else:
                break

        parity_streak = 1
        for j in range(idx - 1, idx - window, -1):
            if parities[j] == parities[j - 1]:
                parity_streak += 1
            else:
                break

        number_counts = Counter(prev_numbers)
        most_common_number, most_common_count = number_counts.most_common(1)[0]

        record = {
            "current_round": int(df.iloc[idx]["round"]),
            "current_number": numbers[idx],
            "current_color": colors[idx],
            "color_streak": color_streak,
            "parity_streak": parity_streak,
            "window_entropy": shannon_entropy(prev_numbers),
            "red_ratio_window": prev_colors.count("red") / window,
            "black_ratio_window": prev_colors.count("black") / window,
            "zero_ratio_window": prev_colors.count("green") / window,
            "high_ratio_window": prev_ranges.count("high") / window,
            "low_ratio_window": prev_ranges.count("low") / window,
            "most_common_number_window": most_common_number,
            "most_common_count_window": most_common_count,
            "next_number": numbers[idx + 1],
            "next_color": colors[idx + 1],
        }

        for step in range(1, window + 1):
            record[f"prev_number_{step}"] = numbers[idx - step]
            record[f"prev_color_{step}"] = colors[idx - step]
            record[f"prev_parity_{step}"] = parities[idx - step]
            record[f"prev_range_{step}"] = ranges[idx - step]

        records.append(record)

    feature_df = pd.DataFrame(records)
    target = feature_df.pop("next_number").astype(int)
    feature_df["next_color_label"] = feature_df.pop("next_color")
    return feature_df, target


def engineer_poker_features(
    action_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Aggregate poker action logs into player-hand modeling features."""
    action_df = action_df.copy()
    summary_df = summary_df.copy()

    action_df["is_aggressive"] = action_df["action"].isin(["raises", "bets", "allin", "caps"]).astype(int)
    action_df["is_passive"] = action_df["action"].isin(["calls", "checks"]).astype(int)
    action_df["is_voluntary_preflop"] = (
        action_df["stage"].eq("preflop")
        & action_df["action"].isin(["calls", "raises", "bets", "allin", "caps"])
    ).astype(int)
    action_df["is_preflop_raise"] = (
        action_df["stage"].eq("preflop")
        & action_df["action"].isin(["raises", "allin", "caps"])
    ).astype(int)

    action_counts = (
        action_df.pivot_table(
            index=["game_id", "player"],
            columns="action",
            values="stage",
            aggfunc="count",
            fill_value=0,
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )
    action_counts.columns = [col if isinstance(col, str) else str(col) for col in action_counts.columns]

    stage_counts = (
        action_df.pivot_table(
            index=["game_id", "player"],
            columns="stage",
            values="action",
            aggfunc="count",
            fill_value=0,
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )
    stage_counts.columns = [col if isinstance(col, str) else str(col) for col in stage_counts.columns]

    grouped = (
        action_df.groupby(["game_id", "player"], as_index=False)
        .agg(
            seat=("seat", "first"),
            position=("position", "first"),
            stack_size=("stack_size", "first"),
            big_blind=("big_blind", "first"),
            small_blind=("small_blind", "first"),
            action_count=("action", "count"),
            avg_action_amount=("bet_amount", "mean"),
            max_action_amount=("bet_amount", "max"),
            aggressive_actions=("is_aggressive", "sum"),
            passive_actions=("is_passive", "sum"),
            vpip=("is_voluntary_preflop", "max"),
            pfr=("is_preflop_raise", "max"),
        )
    )

    poker_df = (
        grouped.merge(action_counts, on=["game_id", "player"], how="left")
        .merge(stage_counts, on=["game_id", "player"], how="left")
        .merge(summary_df, on=["game_id", "player"], how="left", suffixes=("", "_summary"))
    )

    poker_df = poker_df.fillna(0)
    poker_df["stack_in_bb"] = poker_df["stack_size"] / poker_df["big_blind"].replace(0, np.nan)
    poker_df["actual_total_bet"] = poker_df["bets"]
    poker_df["profit_bb"] = poker_df["profit"] / poker_df["big_blind"].replace(0, np.nan)
    poker_df["total_bet_bb"] = poker_df["actual_total_bet"] / poker_df["big_blind"].replace(0, np.nan)
    poker_df["pot_odds_approx"] = poker_df["actual_total_bet"] / poker_df["pot_amount"].replace(0, np.nan)
    poker_df["bet_to_stack_ratio"] = poker_df["actual_total_bet"] / poker_df["stack_size"].replace(0, np.nan)
    poker_df["aggression_score"] = poker_df["aggressive_actions"] / (poker_df["passive_actions"] + 1.0)
    poker_df["position_score"] = poker_df["position"].map(
        {
            "btn": 1.0,
            "co": 0.7,
            "late": 0.55,
            "middle": 0.15,
            "early": -0.2,
            "utg": -0.35,
            "sb": -0.55,
            "bb": -0.45,
            "btn_sb": 0.3,
            "unknown": 0.0,
        }
    ).fillna(0.0)
    poker_df["won_hand"] = (poker_df["profit"] > 0).astype(int)
    poker_df = poker_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_columns = [
        "seat",
        "position",
        "stack_size",
        "stack_in_bb",
        "big_blind",
        "small_blind",
        "action_count",
        "avg_action_amount",
        "max_action_amount",
        "aggressive_actions",
        "passive_actions",
        "vpip",
        "pfr",
        "actual_total_bet",
        "total_bet_bb",
        "pot_odds_approx",
        "bet_to_stack_ratio",
        "aggression_score",
        "position_score",
        "went_to_showdown",
        "showed_cards",
        "mucked",
        "blind",
        "preflop",
        "flop",
        "turn",
        "river",
        "folds",
        "calls",
        "raises",
        "checks",
        "bets",
        "allin",
        "caps",
    ]
    feature_columns = [column for column in feature_columns if column in poker_df.columns]
    feature_df = poker_df[["game_id", "player"] + feature_columns].copy()
    target_classification = poker_df["won_hand"].astype(int)
    target_regression = poker_df["profit_bb"].astype(float)

    return feature_df, target_classification, target_regression, poker_df


def cluster_poker_players(poker_df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Cluster poker players by aggregate behavior for visualization."""
    player_style_df = (
        poker_df.groupby("player", as_index=False)
        .agg(
            hands_played=("game_id", "count"),
            avg_profit_bb=("profit_bb", "mean"),
            avg_aggression=("aggression_score", "mean"),
            avg_vpip=("vpip", "mean"),
            avg_pfr=("pfr", "mean"),
            avg_position_score=("position_score", "mean"),
            showdown_rate=("went_to_showdown", "mean"),
        )
    )

    scaled_features = StandardScaler().fit_transform(
        player_style_df[
            [
                "hands_played",
                "avg_profit_bb",
                "avg_aggression",
                "avg_vpip",
                "avg_pfr",
                "avg_position_score",
                "showdown_rate",
            ]
        ]
    )
    model = KMeans(n_clusters=4, random_state=random_state, n_init="auto")
    player_style_df["cluster"] = model.fit_predict(scaled_features)
    return player_style_df
