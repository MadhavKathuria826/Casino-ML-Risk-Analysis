"""Model training utilities for the casino ML project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluation import evaluate_classification_model, evaluate_regression_model
from utils import DEFAULT_RANDOM_STATE, ensure_dir


def _split_columns(feature_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [column for column in feature_df.columns if column not in categorical_columns]
    return numeric_columns, categorical_columns


def build_preprocessor(
    feature_df: pd.DataFrame,
    use_numeric_pca: bool = False,
    pca_variance_ratio: float = 0.95,
) -> ColumnTransformer:
    """Create a reusable preprocessing transformer."""
    numeric_columns, categorical_columns = _split_columns(feature_df)

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if use_numeric_pca and numeric_columns:
        numeric_steps.append(("pca", PCA(n_components=pca_variance_ratio, random_state=DEFAULT_RANDOM_STATE)))

    numeric_transformer = Pipeline(numeric_steps)
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )


def train_blackjack_models(
    feature_df: pd.DataFrame,
    target: pd.Series,
    output_dir: str | Path,
    use_pca: bool = False,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[dict[str, Any]]:
    """Train blackjack classifiers and persist the fitted pipelines."""
    output_dir = ensure_dir(output_dir)
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df,
        target,
        test_size=0.2,
        stratify=target,
        random_state=random_state,
    )

    models = {
        "blackjack_logistic_regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "blackjack_random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=4,
            n_jobs=1,
            random_state=random_state,
        ),
    }

    results: list[dict[str, Any]] = []
    for name, estimator in models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(feature_df, use_numeric_pca=use_pca)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        result = evaluate_classification_model(name, "blackjack", pipeline, X_test, y_test)
        joblib.dump(pipeline, Path(output_dir) / f"{name}.joblib")
        results.append(result)
    return results


def train_roulette_models(
    feature_df: pd.DataFrame,
    target: pd.Series,
    output_dir: str | Path,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[dict[str, Any]]:
    """Train roulette prediction models against a random baseline."""
    output_dir = ensure_dir(output_dir)
    split_index = int(len(feature_df) * 0.8)

    X_train = feature_df.iloc[:split_index].drop(columns=["next_color_label"])
    X_test = feature_df.iloc[split_index:].drop(columns=["next_color_label"])
    y_train = target.iloc[:split_index]
    y_test = target.iloc[split_index:]

    models = {
        "roulette_dummy_uniform": DummyClassifier(strategy="uniform", random_state=random_state),
        "roulette_random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=1,
            random_state=random_state,
        ),
    }

    results: list[dict[str, Any]] = []
    for name, estimator in models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        result = evaluate_classification_model(name, "roulette", pipeline, X_test, y_test)
        joblib.dump(pipeline, Path(output_dir) / f"{name}.joblib")
        results.append(result)
    return results


def train_poker_models(
    feature_df: pd.DataFrame,
    target_classification: pd.Series,
    target_regression: pd.Series,
    output_dir: str | Path,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> list[dict[str, Any]]:
    """Train poker win-probability and profit models with game-aware splits."""
    output_dir = ensure_dir(output_dir)
    groups = feature_df["game_id"]
    model_features = feature_df.drop(columns=["game_id", "player"])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(model_features, target_classification, groups=groups))

    X_train = model_features.iloc[train_idx]
    X_test = model_features.iloc[test_idx]
    y_train_cls = target_classification.iloc[train_idx]
    y_test_cls = target_classification.iloc[test_idx]
    y_train_reg = target_regression.iloc[train_idx]
    y_test_reg = target_regression.iloc[test_idx]

    classification_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(model_features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=16,
                    min_samples_leaf=3,
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    classification_pipeline.fit(X_train, y_train_cls)
    classification_result = evaluate_classification_model(
        "poker_random_forest_classifier",
        "poker",
        classification_pipeline,
        X_test,
        y_test_cls,
    )
    joblib.dump(classification_pipeline, Path(output_dir) / "poker_random_forest_classifier.joblib")

    results: list[dict[str, Any]] = [classification_result]
    regressors = {
        "poker_gradient_boosting_regressor": GradientBoostingRegressor(random_state=random_state),
        "poker_random_forest_regressor": RandomForestRegressor(
            n_estimators=250,
            max_depth=15,
            min_samples_leaf=3,
            n_jobs=1,
            random_state=random_state,
        ),
    }

    for name, estimator in regressors.items():
        regression_pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(model_features)),
                ("model", estimator),
            ]
        )
        regression_pipeline.fit(X_train, y_train_reg)
        result = evaluate_regression_model(name, "poker", regression_pipeline, X_test, y_test_reg)
        joblib.dump(regression_pipeline, Path(output_dir) / f"{name}.joblib")
        results.append(result)

    return results
