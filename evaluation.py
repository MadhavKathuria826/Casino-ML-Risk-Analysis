"""Evaluation and visualization helpers for the casino ML project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from utils import ensure_dir, save_json


sns.set_theme(style="whitegrid", palette="deep")


def evaluate_classification_model(
    model_name: str,
    game: str,
    fitted_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate a classifier and return a serializable result bundle."""
    y_pred = fitted_model.predict(X_test)
    return {
        "model_name": model_name,
        "game": game,
        "task_type": "classification",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "support": int(len(y_test)),
        "y_true": y_test.tolist(),
        "y_pred": pd.Series(y_pred).tolist(),
    }


def evaluate_regression_model(
    model_name: str,
    game: str,
    fitted_model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate a regressor and return a serializable result bundle."""
    y_pred = fitted_model.predict(X_test)
    return {
        "model_name": model_name,
        "game": game,
        "task_type": "regression",
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": r2_score(y_test, y_pred),
        "support": int(len(y_test)),
        "y_true": y_test.tolist(),
        "y_pred": pd.Series(y_pred).tolist(),
    }


def save_metrics_tables(
    model_results: list[dict[str, Any]],
    simulation_results: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist flat metrics tables for models and simulations."""
    output_dir = ensure_dir(output_dir)

    metrics_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for result in model_results:
        flat_row = {
            key: value
            for key, value in result.items()
            if key not in {"confusion_matrix", "y_true", "y_pred"}
        }
        metrics_rows.append(flat_row)

        if result["task_type"] == "classification":
            comparison_rows.append(
                {
                    "game": result["game"],
                    "model_name": result["model_name"],
                    "primary_metric": "accuracy",
                    "metric_value": result["accuracy"],
                }
            )
        else:
            comparison_rows.append(
                {
                    "game": result["game"],
                    "model_name": result["model_name"],
                    "primary_metric": "rmse",
                    "metric_value": result["rmse"],
                }
            )

    for game, game_results in simulation_results.items():
        for strategy, metrics in game_results.items():
            metrics_rows.append(
                {
                    "game": game,
                    "model_name": strategy,
                    "task_type": "simulation",
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "ev": metrics["ev"],
                    "variance": metrics["variance"],
                    "risk_of_ruin": metrics["risk_of_ruin"],
                    "support": metrics["support"],
                }
            )
            comparison_rows.append(
                {
                    "game": game,
                    "model_name": strategy,
                    "primary_metric": "ev",
                    "metric_value": metrics["ev"],
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    comparisons_df = pd.DataFrame(comparison_rows)

    metrics_path = output_dir / "metrics_summary.csv"
    comparisons_path = output_dir / "model_comparisons.csv"
    metrics_df.to_csv(metrics_path, index=False)
    comparisons_df.to_csv(comparisons_path, index=False)
    save_json(simulation_results, output_dir / "simulation_results.json")
    return metrics_path, comparisons_path


def plot_accuracy_comparison(model_results: list[dict[str, Any]], figures_dir: str | Path) -> Path:
    """Plot classification accuracy comparisons across games."""
    figures_dir = ensure_dir(figures_dir)
    plot_df = pd.DataFrame(
        [result for result in model_results if result["task_type"] == "classification"]
    )[["game", "model_name", "accuracy"]]

    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="game", y="accuracy", hue="model_name")
    plt.title("Classification Accuracy Across Casino Games")
    plt.ylabel("Accuracy")
    plt.xlabel("Game")
    plt.tight_layout()
    output_path = Path(figures_dir) / "accuracy_comparison.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_roi_over_time(simulation_results: dict[str, Any], figures_dir: str | Path) -> Path:
    """Plot representative bankroll trajectories for each simulation strategy."""
    figures_dir = ensure_dir(figures_dir)
    plt.figure(figsize=(12, 7))

    for game, strategies in simulation_results.items():
        for strategy, metrics in strategies.items():
            trace = metrics.get("median_trace", [])
            if trace:
                plt.plot(trace, label=f"{game}:{strategy}")

    plt.title("ROI / Bankroll Evolution Over Time")
    plt.xlabel("Step")
    plt.ylabel("Bankroll")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_path = Path(figures_dir) / "roi_over_time.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_winnings_distribution(simulation_results: dict[str, Any], figures_dir: str | Path) -> Path:
    """Plot end-of-session winnings distributions for simulation strategies."""
    figures_dir = ensure_dir(figures_dir)
    rows: list[dict[str, Any]] = []
    for game, strategies in simulation_results.items():
        for strategy, metrics in strategies.items():
            for value in metrics.get("terminal_outcomes", []):
                rows.append({"game": game, "strategy": strategy, "terminal_profit": value})

    distribution_df = pd.DataFrame(rows)
    plt.figure(figsize=(12, 7))
    sns.histplot(
        data=distribution_df,
        x="terminal_profit",
        hue="strategy",
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("Distribution of Simulated Winnings")
    plt.xlabel("Terminal Profit")
    plt.tight_layout()
    output_path = Path(figures_dir) / "winnings_distribution.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_roulette_failure(model_results: list[dict[str, Any]], figures_dir: str | Path) -> Path:
    """Visualize roulette prediction failure versus a random baseline."""
    figures_dir = ensure_dir(figures_dir)
    roulette_df = pd.DataFrame(
        [result for result in model_results if result["game"] == "roulette" and result["task_type"] == "classification"]
    )[["model_name", "accuracy"]]

    plt.figure(figsize=(8, 5))
    sns.barplot(data=roulette_df, x="model_name", y="accuracy")
    plt.axhline(1 / 37, color="red", linestyle="--", label="Pure random baseline (1/37)")
    plt.title("Roulette Prediction Accuracy vs Random Baseline")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    output_path = Path(figures_dir) / "roulette_randomness.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_poker_clusters(player_style_df: pd.DataFrame, figures_dir: str | Path) -> Path:
    """Plot poker player behavior clusters."""
    figures_dir = ensure_dir(figures_dir)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=player_style_df,
        x="avg_aggression",
        y="avg_profit_bb",
        hue="cluster",
        size="hands_played",
        sizes=(30, 250),
    )
    plt.title("Poker Player Behavior Clusters")
    plt.xlabel("Average Aggression Score")
    plt.ylabel("Average Profit (BB)")
    plt.tight_layout()
    output_path = Path(figures_dir) / "poker_behavior_clusters.png"
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
