import json
import time
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATASET_PATH = Path(__file__).with_name("creditcard.csv")
RESULT_PATH = Path(__file__).with_name("benchmark_result.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CLASSIFICATION_THRESHOLD = 0.5

LATENCY_WARMUP_RUNS = 20
LATENCY_MEASURE_RUNS = 200

BATCH_SIZE = 1000
BATCH_WARMUP_RUNS = 5
BATCH_MEASURE_RUNS = 30


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {DATASET_PATH}")

    print(f"Đang đọc dataset: {DATASET_PATH}")
    load_start = time.perf_counter()
    data = pd.read_csv(DATASET_PATH)
    load_time_seconds = time.perf_counter() - load_start

    if "Class" not in data.columns:
        raise ValueError("Dataset không có cột mục tiêu 'Class'")

    X = data.drop(columns=["Class"])
    y = data["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_parameters = {
        "objective": "binary",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }

    print("Đang train LGBMClassifier...")
    model = LGBMClassifier(**model_parameters)

    training_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time_seconds = time.perf_counter() - training_start

    print("Đang tính metrics...")
    fraud_probabilities = model.predict_proba(X_test)[:, 1]
    fraud_predictions = (
        fraud_probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    auc_roc = roc_auc_score(y_test, fraud_probabilities)
    accuracy = accuracy_score(y_test, fraud_predictions)
    precision = precision_score(
        y_test, fraud_predictions, zero_division=0
    )
    recall = recall_score(
        y_test, fraud_predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, fraud_predictions, zero_division=0
    )

    # Warm-up trước khi đo latency một dòng.
    one_row = X_test.iloc[[0]]
    for _ in range(LATENCY_WARMUP_RUNS):
        model.predict_proba(one_row)

    latency_start = time.perf_counter()
    for _ in range(LATENCY_MEASURE_RUNS):
        model.predict_proba(one_row)
    latency_elapsed = time.perf_counter() - latency_start

    inference_latency_ms_one_row = (
        latency_elapsed / LATENCY_MEASURE_RUNS
    ) * 1000.0

    # Warm-up và đo prediction batch đúng 1.000 dòng.
    batch = X_test.iloc[:BATCH_SIZE]

    if len(batch) != BATCH_SIZE:
        raise ValueError(
            f"Test set chỉ có {len(batch)} dòng, không đủ batch {BATCH_SIZE}"
        )

    for _ in range(BATCH_WARMUP_RUNS):
        model.predict_proba(batch)

    batch_start = time.perf_counter()
    for _ in range(BATCH_MEASURE_RUNS):
        model.predict_proba(batch)
    batch_elapsed = time.perf_counter() - batch_start

    average_batch_time_ms = (
        batch_elapsed / BATCH_MEASURE_RUNS
    ) * 1000.0

    inference_throughput_rows_per_second = (
        BATCH_SIZE * BATCH_MEASURE_RUNS
    ) / batch_elapsed

    # best_iteration_ chỉ có ý nghĩa khi dùng early stopping.
    best_iteration = getattr(model, "best_iteration_", None)
    if not best_iteration:
        best_iteration = None
    else:
        best_iteration = int(best_iteration)

    result = {
        "cloud": "gcp",
        "instance_type": "e2-medium",
        "dataset_rows": int(len(data)),
        "dataset_features": int(X.shape[1]),
        "fraud_rows": int(y.sum()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size_ratio": TEST_SIZE,
        "model": "LGBMClassifier",
        "model_parameters": model_parameters,
        "best_iteration": best_iteration,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "load_time_seconds": float(load_time_seconds),
        "training_time_seconds": float(training_time_seconds),
        "auc_roc": float(auc_roc),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "latency_warmup_runs": LATENCY_WARMUP_RUNS,
        "latency_measure_runs": LATENCY_MEASURE_RUNS,
        "inference_latency_ms_one_row": float(
            inference_latency_ms_one_row
        ),
        "inference_batch_size_rows": BATCH_SIZE,
        "inference_batch_time_ms": float(average_batch_time_ms),
        "inference_throughput_rows_per_second": float(
            inference_throughput_rows_per_second
        ),
    }

    RESULT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nBenchmark hoàn thành:")
    print(json.dumps(result, indent=2))
    print(f"\nĐã lưu kết quả tại: {RESULT_PATH}")


if __name__ == "__main__":
    main()
