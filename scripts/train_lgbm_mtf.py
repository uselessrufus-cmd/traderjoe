import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


def train_model(df: pd.DataFrame):
    # Drop non-numeric columns except label
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in {"label"}]

    # Remove timestamps from features
    feature_cols = [c for c in feature_cols if not c.endswith("timestamp")]

    X = df[feature_cols]
    y = df["label"]

    # Train/valid split (last 20% for validation)
    split = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "metric": "multi_logloss",
        "verbosity": -1,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train + 1)
    lgb_val = lgb.Dataset(X_val, label=y_val + 1)

    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    preds = model.predict(X_val)
    pred_labels = np.argmax(preds, axis=1) - 1
    acc = float((pred_labels == y_val.values).mean())

    return model, acc, feature_cols


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/ai_memory.db")
    p.add_argument("--model", default="data/models/lgbm_mtf_model.txt")
    p.add_argument("--meta", default="data/models/lgbm_mtf_meta.txt")
    args = p.parse_args()

    with sqlite3.connect(args.db) as con:
        df = pd.read_sql_query("SELECT * FROM features", con)

    model, acc, feature_cols = train_model(df)

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model)

    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    # Save feature list in model for later prediction
    model.save_model(args.model, num_iteration=model.best_iteration)
    Path(args.meta).write_text(
        "\n".join(feature_cols) + f"\nvalidation_accuracy={acc:.4f}\n",
        encoding="ascii",
    )

    print(f"Saved model to {args.model}")
    print(f"Validation accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()
