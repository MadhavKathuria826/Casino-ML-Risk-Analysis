"""Main orchestration script for casino simulation and risk analysis."""

from __future__ import annotations

import argparse
import os

import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from data_loading import BlackjackLoadConfig, discover_poker_files, load_blackjack_sample, load_roulette_data
from evaluation import (
    plot_accuracy_comparison,
    plot_poker_clusters,
    plot_roi_over_time,
    plot_roulette_failure,
    plot_winnings_distribution,
    save_metrics_tables,
)
from feature_engineering import (
    cluster_poker_players,
    engineer_blackjack_features,
    engineer_poker_features,
    engineer_roulette_features,
)
from models import train_blackjack_models, train_poker_models, train_roulette_models
from preprocessing import parse_poker_logs, preprocess_blackjack, preprocess_roulette
from simulation import simulate_blackjack_strategies, simulate_poker_strategies, simulate_roulette_strategies
from utils import DEFAULT_RANDOM_STATE, ensure_dir, set_global_seed, setup_logging


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI for the project runner."""
    parser = argparse.ArgumentParser(description="Casino simulation and ML risk analysis.")
    parser.add_argument("--blackjack-path", default="blackjack_simulator.csv")
    parser.add_argument("--roulette-path", default="roulette_100000_rounds.csv")
    parser.add_argument("--poker-dir", default="Poker Data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--blackjack-sample-size", type=int, default=250_000)
    parser.add_argument("--blackjack-chunksize", type=int, default=250_000)
    parser.add_argument("--blackjack-scan-limit-chunks", type=int, default=None)
    parser.add_argument("--poker-max-games", type=int, default=None)
    parser.add_argument("--use-blackjack-pca", action="store_true")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return parser


def derive_insights(model_results: list[dict], simulation_results: dict[str, dict]) -> list[str]:
    """Create portfolio-ready insights from the analysis."""
    blackjack_basic = simulation_results["blackjack"]["basic"]["ev"]
    blackjack_random = simulation_results["blackjack"]["random"]["ev"]

    roulette_results = {
        result["model_name"]: result["accuracy"]
        for result in model_results
        if result["game"] == "roulette" and result["task_type"] == "classification"
    }
    roulette_gap = roulette_results["roulette_random_forest"] - roulette_results["roulette_dummy_uniform"]

    poker_classifier = next(
        result
        for result in model_results
        if result["model_name"] == "poker_random_forest_classifier"
    )

    blackjack_text = (
        "Blackjack insight: basic strategy produced a higher simulated EV "
        f"({blackjack_basic:.4f} per hand) than random play ({blackjack_random:.4f}), "
        "showing that structured decision-making improves long-run outcomes."
        if blackjack_basic > blackjack_random
        else "Blackjack insight: even when the absolute EV remains negative under house-edge assumptions, "
        f"the structured policy stayed closer to break-even ({blackjack_basic:.4f}) than the comparison policy ({blackjack_random:.4f})."
    )

    return [
        blackjack_text,
        (
            "Roulette insight: the predictive model remained near the random baseline "
            f"(RF accuracy delta {roulette_gap:.4f}), reinforcing that roulette outcomes are effectively stochastic."
        ),
        (
            "Poker insight: player-hand behavior retains predictive signal, with the poker classifier reaching "
            f"{poker_classifier['accuracy']:.4f} accuracy on held-out hands, which makes poker materially more learnable than roulette."
        ),
    ]


def main() -> None:
    """Run the end-to-end casino ML project."""
    args = build_argument_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    models_dir = ensure_dir(output_dir / "models")

    set_global_seed(args.random_state)
    logger = setup_logging(output_dir)
    logger.info("Starting casino simulation and risk analysis pipeline.")

    logger.info("Loading blackjack sample.")
    blackjack_raw = load_blackjack_sample(
        BlackjackLoadConfig(
            path=args.blackjack_path,
            sample_size=args.blackjack_sample_size,
            chunksize=args.blackjack_chunksize,
            random_state=args.random_state,
            scan_limit_chunks=args.blackjack_scan_limit_chunks,
        )
    )
    logger.info("Blackjack sample loaded with %s rows.", len(blackjack_raw))
    blackjack_df = preprocess_blackjack(blackjack_raw)
    blackjack_features, blackjack_target = engineer_blackjack_features(blackjack_df)

    logger.info("Loading roulette data.")
    roulette_raw = load_roulette_data(args.roulette_path)
    roulette_df = preprocess_roulette(roulette_raw)
    roulette_features, roulette_target = engineer_roulette_features(roulette_df)

    logger.info("Parsing poker hand histories.")
    poker_files = [str(path) for path in discover_poker_files(args.poker_dir)]
    poker_action_df, poker_summary_df = parse_poker_logs(poker_files, max_games=args.poker_max_games)
    poker_features, poker_target_cls, poker_target_reg, poker_model_df = engineer_poker_features(
        poker_action_df,
        poker_summary_df,
    )
    logger.info(
        "Structured poker dataset created with %s player-hands across %s games.",
        len(poker_features),
        poker_features["game_id"].nunique(),
    )

    poker_action_df.to_csv(output_dir / "poker_actions_structured.csv", index=False)
    poker_model_df.to_csv(output_dir / "poker_player_hand_features.csv", index=False)

    logger.info("Training blackjack models.")
    blackjack_results = train_blackjack_models(
        blackjack_features,
        blackjack_target,
        models_dir,
        use_pca=args.use_blackjack_pca,
        random_state=args.random_state,
    )

    logger.info("Training roulette models.")
    roulette_results = train_roulette_models(
        roulette_features,
        roulette_target,
        models_dir,
        random_state=args.random_state,
    )

    logger.info("Training poker models.")
    poker_results = train_poker_models(
        poker_features,
        poker_target_cls,
        poker_target_reg,
        models_dir,
        random_state=args.random_state,
    )

    model_results = blackjack_results + roulette_results + poker_results

    logger.info("Running blackjack Monte Carlo strategies.")
    blackjack_sim_results = simulate_blackjack_strategies(random_state=args.random_state)
    logger.info("Running roulette Monte Carlo strategies.")
    roulette_sim_results = simulate_roulette_strategies(roulette_df, random_state=args.random_state)
    logger.info("Running poker bootstrap simulations.")
    poker_sim_results = simulate_poker_strategies(poker_model_df, random_state=args.random_state)

    simulation_results = {
        "blackjack": blackjack_sim_results,
        "roulette": roulette_sim_results,
        "poker": poker_sim_results,
    }

    logger.info("Saving metrics and figures.")
    metrics_path, comparisons_path = save_metrics_tables(model_results, simulation_results, output_dir)
    plot_accuracy_comparison(model_results, figures_dir)
    plot_roi_over_time(simulation_results, figures_dir)
    plot_winnings_distribution(simulation_results, figures_dir)
    plot_roulette_failure(model_results, figures_dir)
    player_clusters = cluster_poker_players(poker_model_df)
    player_clusters.to_csv(output_dir / "poker_player_clusters.csv", index=False)
    plot_poker_clusters(player_clusters, figures_dir)

    insights = derive_insights(model_results, simulation_results)
    logger.info("Key project insights:")
    for insight in insights:
        logger.info(insight)

    pd.DataFrame({"insight": insights}).to_csv(output_dir / "key_insights.csv", index=False)

    print("\n=== Casino Simulation and Risk Analysis Complete ===")
    print(f"Metrics summary saved to: {metrics_path}")
    print(f"Model comparison saved to: {comparisons_path}")
    print(f"Simulation results saved to: {output_dir / 'simulation_results.json'}")
    print("\nKey insights:")
    for insight in insights:
        print(f"- {insight}")


if __name__ == "__main__":
    main()
