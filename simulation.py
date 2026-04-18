"""Monte Carlo simulation components for casino risk analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils import DEFAULT_RANDOM_STATE


@dataclass(slots=True)
class BlackjackState:
    """Blackjack hand state."""

    player_cards: list[int]
    dealer_cards: list[int]
    bet: float = 1.0
    doubled: bool = False


CARD_SHOE = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11])


def blackjack_hand_value(cards: list[int]) -> tuple[int, bool]:
    """Compute blackjack hand total and softness."""
    total = sum(cards)
    aces = cards.count(11)
    soft = aces > 0
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total, soft and total <= 21


def draw_card(rng: np.random.Generator) -> int:
    """Draw a card from an infinite shoe."""
    return int(rng.choice(CARD_SHOE))


def basic_strategy_action(player_cards: list[int], dealer_up: int, can_double: bool) -> str:
    """A concise approximation of blackjack basic strategy."""
    total, soft = blackjack_hand_value(player_cards)

    if soft:
        if total <= 17:
            return "hit"
        if total == 18:
            if dealer_up in {3, 4, 5, 6} and can_double:
                return "double"
            if dealer_up in {9, 10, 11}:
                return "hit"
            return "stand"
        return "stand"

    if total <= 8:
        return "hit"
    if total == 9:
        return "double" if can_double and dealer_up in {3, 4, 5, 6} else "hit"
    if total == 10:
        return "double" if can_double and dealer_up in {2, 3, 4, 5, 6, 7, 8, 9} else "hit"
    if total == 11:
        return "double" if can_double and dealer_up != 11 else "hit"
    if total == 12:
        return "stand" if dealer_up in {4, 5, 6} else "hit"
    if total in {13, 14, 15, 16}:
        return "stand" if dealer_up in {2, 3, 4, 5, 6} else "hit"
    return "stand"


def random_strategy_action(player_cards: list[int], can_double: bool, rng: np.random.Generator) -> str:
    """Random blackjack policy for comparison."""
    actions = ["hit", "stand"]
    if can_double:
        actions.append("double")
    return str(rng.choice(actions))


def resolve_blackjack_hand(strategy_name: str, rng: np.random.Generator) -> float:
    """Play a simplified blackjack hand and return profit."""
    state = BlackjackState(
        player_cards=[draw_card(rng), draw_card(rng)],
        dealer_cards=[draw_card(rng), draw_card(rng)],
    )

    player_total, _ = blackjack_hand_value(state.player_cards)
    dealer_total, _ = blackjack_hand_value(state.dealer_cards)
    player_blackjack = len(state.player_cards) == 2 and player_total == 21
    dealer_blackjack = len(state.dealer_cards) == 2 and dealer_total == 21

    if player_blackjack and dealer_blackjack:
        return 0.0
    if player_blackjack:
        return 1.5
    if dealer_blackjack:
        return -1.0

    while True:
        can_double = len(state.player_cards) == 2 and not state.doubled
        if strategy_name == "basic":
            action = basic_strategy_action(state.player_cards, state.dealer_cards[0], can_double)
        else:
            action = random_strategy_action(state.player_cards, can_double, rng)

        if action == "stand":
            break
        if action == "double":
            state.bet *= 2
            state.doubled = True
            state.player_cards.append(draw_card(rng))
            player_total, _ = blackjack_hand_value(state.player_cards)
            if player_total > 21:
                return -state.bet
            break

        state.player_cards.append(draw_card(rng))
        player_total, _ = blackjack_hand_value(state.player_cards)
        if player_total > 21:
            return -state.bet

    while True:
        dealer_total, _ = blackjack_hand_value(state.dealer_cards)
        if dealer_total >= 17:
            break
        state.dealer_cards.append(draw_card(rng))

    player_total, _ = blackjack_hand_value(state.player_cards)
    dealer_total, _ = blackjack_hand_value(state.dealer_cards)

    if dealer_total > 21 or player_total > dealer_total:
        return state.bet
    if player_total < dealer_total:
        return -state.bet
    return 0.0


def summarize_traces(traces: list[list[float]], terminal_outcomes: list[float], profits: list[float]) -> dict[str, object]:
    """Summarize simulation outputs into portfolio-friendly metrics."""
    median_trace = np.median(np.array([np.array(trace) for trace in traces]), axis=0).tolist()
    return {
        "ev": float(np.mean(profits)),
        "variance": float(np.var(profits)),
        "risk_of_ruin": float(np.mean([terminal <= 0 for terminal in terminal_outcomes])),
        "support": int(len(profits)),
        "median_trace": median_trace,
        "terminal_outcomes": [float(value) for value in terminal_outcomes[:500]],
    }


def simulate_blackjack_strategies(
    hands_per_trial: int = 300,
    n_trials: int = 400,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, dict[str, object]]:
    """Compare basic and random blackjack strategies by Monte Carlo simulation."""
    results: dict[str, dict[str, object]] = {}
    for offset, strategy_name in enumerate(["basic", "random"]):
        rng = np.random.default_rng(random_state + offset)
        traces: list[list[float]] = []
        terminal_outcomes: list[float] = []
        profits: list[float] = []

        for _ in range(n_trials):
            bankroll = 0.0
            trace = [bankroll]
            for _ in range(hands_per_trial):
                bankroll += resolve_blackjack_hand(strategy_name, rng)
                trace.append(bankroll)
            traces.append(trace)
            terminal_outcomes.append(bankroll)
            profits.append(bankroll / hands_per_trial)

        results[strategy_name] = summarize_traces(traces, terminal_outcomes, profits)
    return results


def roulette_spin(rng: np.random.Generator, number_probs: np.ndarray, numbers: np.ndarray) -> int:
    """Sample a roulette number from empirical probabilities."""
    return int(rng.choice(numbers, p=number_probs))


def roulette_even_money_payout(number: int, color_lookup: dict[int, str], target_color: str = "red") -> float:
    """Return even-money roulette payoff for a color bet."""
    return 1.0 if color_lookup[number] == target_color else -1.0


def simulate_roulette_strategies(
    roulette_df: pd.DataFrame,
    spins_per_session: int = 300,
    n_sessions: int = 500,
    initial_bankroll: float = 200.0,
    base_bet: float = 1.0,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, dict[str, object]]:
    """Simulate flat betting and Martingale on roulette using empirical wheel probabilities."""
    number_distribution = roulette_df["winning_number"].value_counts(normalize=True).sort_index()
    numbers = number_distribution.index.to_numpy()
    number_probs = number_distribution.to_numpy()
    color_lookup = roulette_df.drop_duplicates("winning_number").set_index("winning_number")["winning_color"].to_dict()

    def run_session(strategy: str, rng: np.random.Generator) -> list[float]:
        bankroll = initial_bankroll
        current_bet = base_bet
        trace = [bankroll]

        for _ in range(spins_per_session):
            if bankroll <= 0:
                trace.append(bankroll)
                continue

            wager = min(current_bet, bankroll)
            bankroll -= wager
            winning_number = roulette_spin(rng, number_probs, numbers)
            payout = roulette_even_money_payout(winning_number, color_lookup, target_color="red")

            if payout > 0:
                bankroll += wager * 2
                current_bet = base_bet
            else:
                current_bet = base_bet if strategy == "flat" else min(current_bet * 2, initial_bankroll)
            trace.append(bankroll)
        return trace

    results: dict[str, dict[str, object]] = {}
    for offset, strategy in enumerate(["flat", "martingale"]):
        rng = np.random.default_rng(random_state + 100 + offset)
        traces = [run_session(strategy, rng) for _ in range(n_sessions)]
        terminal_outcomes = [trace[-1] - initial_bankroll for trace in traces]
        profits = [terminal / spins_per_session for terminal in terminal_outcomes]
        results[strategy] = summarize_traces(traces, terminal_outcomes, profits)
    return results


def simulate_poker_strategies(
    poker_df: pd.DataFrame,
    hands_per_trial: int = 250,
    n_trials: int = 400,
    initial_bankroll_bb: float = 200.0,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, dict[str, object]]:
    """Bootstrap historical poker hands to compare simplified strategy profiles."""
    strategy_masks = {
        "tight_aggressive": (
            (poker_df["vpip"] == 1)
            & (poker_df["pfr"] == 1)
            & (poker_df["aggression_score"] >= poker_df["aggression_score"].median())
            & (poker_df["position_score"] >= 0)
        ),
        "passive_loose": (
            (poker_df["vpip"] == 1)
            & (poker_df["aggression_score"] <= poker_df["aggression_score"].median())
            & (poker_df["position_score"] <= 0.15)
        ),
        "balanced": (poker_df["vpip"] == 1),
    }

    results: dict[str, dict[str, object]] = {}
    for offset, (strategy, mask) in enumerate(strategy_masks.items()):
        subset = poker_df.loc[mask, "profit_bb"]
        if subset.empty:
            continue

        rng = np.random.default_rng(random_state + 200 + offset)
        traces: list[list[float]] = []
        terminal_outcomes: list[float] = []
        profits: list[float] = []

        for _ in range(n_trials):
            sampled = rng.choice(subset.to_numpy(), size=hands_per_trial, replace=True)
            bankroll_path = np.concatenate(([initial_bankroll_bb], initial_bankroll_bb + np.cumsum(sampled)))
            traces.append(bankroll_path.tolist())
            terminal_outcomes.append(float(bankroll_path[-1] - initial_bankroll_bb))
            profits.append(float(np.mean(sampled)))

        results[strategy] = summarize_traces(traces, terminal_outcomes, profits)

    return results
