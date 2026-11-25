"""Compare text classifiers on original and computer-generated product reviews."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import Dataset
from huggingface_hub import hf_hub_download
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


DATASET_ID = "lava5675/Fake-Reviews-Dataset"
DATASET_REVISION = "00f829e21be152e4d5969c0a24b9302cf61cc8bd"
DATASET_FILE = "data/train-00000-of-00001.parquet"
SEED = 42
WORD = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    return " ".join(WORD.findall(unicodedata.normalize("NFKC", text).casefold()))


def prepare_reviews(rows: Dataset) -> tuple[list[str], list[int], dict[str, int]]:
    label_by_text: dict[str, int] = {}
    counts: Counter[str] = Counter()
    conflicts: set[str] = set()
    missing = 0

    for row in rows.select_columns(["text", "label"]):
        text = row["text"]
        label = row["label"]
        if label not in (0, 1):
            raise ValueError(f"Expected a binary label, received {label!r}")
        normalized = normalize_text(text) if isinstance(text, str) else ""
        if not normalized:
            missing += 1
            continue
        counts[normalized] += 1
        previous_label = label_by_text.setdefault(normalized, label)
        if label != previous_label:
            conflicts.add(normalized)

    texts = [text for text in label_by_text if text not in conflicts]
    labels = [label_by_text[text] for text in texts]
    conflicting_rows = sum(counts[text] for text in conflicts)
    return texts, labels, {
        "source_rows": len(rows),
        "empty_text_rows_excluded": missing,
        "conflicting_label_rows_excluded": conflicting_rows,
        "duplicate_rows_excluded": len(rows) - missing - conflicting_rows - len(texts),
        "unique_reviews_kept": len(texts),
    }


def evaluate(model, texts: list[str], labels: list[int]) -> tuple[dict[str, float], list[int]]:
    scores = model.predict_proba(texts)[:, 1]
    predicted = (scores > 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }
    return metrics, predicted.tolist()


def describe_split(labels: list[int]) -> dict[str, int]:
    counts = Counter(labels)
    return {"rows": len(labels), "original": counts[0], "computer_generated": counts[1]}


def run_experiment(rows: Dataset) -> dict:
    texts, labels, preparation = prepare_reviews(rows)
    x_train, x_held, y_train, y_held = train_test_split(
        texts, labels, test_size=0.40, stratify=labels, random_state=SEED,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_held, y_held, test_size=0.50, stratify=y_held, random_state=SEED,
    )

    candidates = {
        "multinomial_nb": MultinomialNB(),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    }
    validation = {}
    fitted_models = {}
    for name, estimator in candidates.items():
        model = make_pipeline(
            TfidfVectorizer(max_features=5000, stop_words="english", min_df=3, max_df=0.8),
            estimator,
        )
        model.fit(x_train, y_train)
        validation[name], _ = evaluate(model, x_val, y_val)
        fitted_models[name] = model
        print(f"{name}: validation F1 = {validation[name]['f1']:.4f}", flush=True)

    selected = max(validation, key=lambda name: validation[name]["f1"])
    model = fitted_models[selected]
    test_metrics, test_predictions = evaluate(model, x_test, y_test)
    return {
        "dataset_revision": DATASET_REVISION,
        "seed": SEED,
        "preparation": preparation,
        "splits": {
            "train": describe_split(y_train),
            "validation": describe_split(y_val),
            "test": describe_split(y_test),
        },
        "label_order": ["original", "computer_generated"],
        "validation": validation,
        "selected_by_validation_f1": selected,
        "test": test_metrics,
        "test_confusion_matrix": confusion_matrix(y_test, test_predictions, labels=[0, 1]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow-file", type=Path, help="Local Arrow cache file for the pinned dataset revision")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    if args.arrow_file is None:
        parquet = hf_hub_download(DATASET_ID, DATASET_FILE, repo_type="dataset", revision=DATASET_REVISION)
        rows = Dataset.from_parquet(parquet, keep_in_memory=True)
    else:
        rows = Dataset.from_file(str(args.arrow_file))
    results = {"dataset": DATASET_ID, **run_experiment(rows)}
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {results['selected_by_validation_f1']}, test F1 = {results['test']['f1']:.4f}")


if __name__ == "__main__":
    main()
