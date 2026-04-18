"""Dataset loading utilities for the casino ML project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from utils import DEFAULT_RANDOM_STATE


@dataclass(slots=True)
class BlackjackLoadConfig:
    """Configuration for chunked blackjack sampling."""

    path: str | Path
    sample_size: int = 250_000
    chunksize: int = 250_000
    random_state: int = DEFAULT_RANDOM_STATE
    scan_limit_chunks: int | None = None


def estimate_csv_rows(path: str | Path, sample_lines: int = 5000) -> int:
    """Estimate row count from file size and average sampled line length."""
    path = Path(path)
    file_size = path.stat().st_size

    sampled_lengths: list[int] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        next(handle, None)
        for _, line in zip(range(sample_lines), handle):
            sampled_lengths.append(len(line.encode("utf-8")))

    if not sampled_lengths:
        return 0

    avg_length = max(np.mean(sampled_lengths), 1)
    estimated_rows = int(file_size / avg_length)
    return max(estimated_rows, len(sampled_lengths))


def load_blackjack_sample(config: BlackjackLoadConfig) -> pd.DataFrame:
    """Load a representative blackjack sample using chunked probabilistic sampling."""
    path = Path(config.path)
    estimated_rows = estimate_csv_rows(path)
    if estimated_rows <= 0:
        raise ValueError(f"Unable to estimate rows for blackjack dataset: {path}")

    effective_rows = estimated_rows
    if config.scan_limit_chunks is not None:
        effective_rows = min(estimated_rows, config.chunksize * config.scan_limit_chunks)

    sample_fraction = min(1.0, (config.sample_size / effective_rows) * 1.2)

    sampled_chunks: list[pd.DataFrame] = []
    for chunk_index, chunk in enumerate(pd.read_csv(path, chunksize=config.chunksize)):
        sampled = chunk.sample(
            frac=sample_fraction,
            random_state=config.random_state + chunk_index,
        )
        sampled_chunks.append(sampled)
        if config.scan_limit_chunks is not None and (chunk_index + 1) >= config.scan_limit_chunks:
            break

    combined = pd.concat(sampled_chunks, ignore_index=True)
    if len(combined) > config.sample_size:
        combined = combined.sample(
            n=config.sample_size,
            random_state=config.random_state,
        ).reset_index(drop=True)
    else:
        combined = combined.reset_index(drop=True)

    return combined


def load_roulette_data(path: str | Path) -> pd.DataFrame:
    """Load the roulette spins dataset."""
    return pd.read_csv(path)


def discover_poker_files(directory: str | Path) -> list[Path]:
    """Discover poker hand history text files in sorted order."""
    directory = Path(directory)
    return sorted(directory.glob("PokerData*.txt"))


def iter_poker_games(paths: list[str | Path], max_games: int | None = None) -> Iterator[list[str]]:
    """Yield individual poker hands split by blank lines or repeated game starts."""
    yielded = 0
    for path_like in paths:
        path = Path(path_like)
        current_game: list[str] = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if line.startswith("Game started at:") and current_game:
                    yield current_game
                    yielded += 1
                    if max_games is not None and yielded >= max_games:
                        return
                    current_game = [line]
                    continue

                if line.strip():
                    current_game.append(line)
                elif current_game:
                    yield current_game
                    yielded += 1
                    if max_games is not None and yielded >= max_games:
                        return
                    current_game = []

        if current_game:
            yield current_game
            yielded += 1
            if max_games is not None and yielded >= max_games:
                return
