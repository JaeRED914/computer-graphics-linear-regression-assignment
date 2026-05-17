#!/usr/bin/env python3
"""Pure linear regression classifier for CIFAR-10 using raw pixel features."""

from __future__ import annotations

import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def unpickle_batch(path: Path) -> dict:
    with path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=np.exceptions.VisibleDeprecationWarning)
            return pickle.load(handle, encoding="bytes")


def load_cifar10(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images = []
    train_labels = []

    for batch_idx in range(1, 6):
        batch = unpickle_batch(data_dir / f"data_batch_{batch_idx}")
        train_images.append(batch[b"data"])
        train_labels.extend(batch[b"labels"])

    test_batch = unpickle_batch(data_dir / "test_batch")

    x_train = np.concatenate(train_images, axis=0).astype(np.float64)
    y_train = np.asarray(train_labels, dtype=np.int64)
    x_test = test_batch[b"data"].astype(np.float64)
    y_test = np.asarray(test_batch[b"labels"], dtype=np.int64)
    return x_train, y_train, x_test, y_test


def add_bias(features: np.ndarray) -> np.ndarray:
    return np.hstack([features, np.ones((features.shape[0], 1), dtype=features.dtype)])


def normalize_with_train_stats(
    train_features: np.ndarray,
    other_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    normalized_train = (train_features - mean) / std
    normalized_other = (other_features - mean) / std
    return normalized_train, normalized_other


def prepare_train_test_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    num_train: int,
    num_test: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_subset = x_train[:num_train] / 255.0
    test_subset = x_test[:num_test] / 255.0
    train_subset, test_subset = normalize_with_train_stats(train_subset, test_subset)
    return add_bias(train_subset), add_bias(test_subset)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def train_linear_regression(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    xtx = x_train.T @ x_train
    xty = x_train.T @ y_train
    try:
        return np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx) @ xty


def predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = x @ weights
    return np.argmax(scores, axis=1)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[truth, pred] += 1
    return matrix


def classification_metrics(matrix: np.ndarray) -> dict:
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_positive = matrix.sum(axis=0).astype(np.float64)
    actual_positive = matrix.sum(axis=1).astype(np.float64)

    precision = np.divide(
        true_positive,
        predicted_positive,
        out=np.zeros_like(true_positive),
        where=predicted_positive != 0,
    )
    recall = np.divide(
        true_positive,
        actual_positive,
        out=np.zeros_like(true_positive),
        where=actual_positive != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    return {
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def format_confusion_matrix(matrix: np.ndarray) -> str:
    header = ["true\\pred"] + [f"{idx:>5}" for idx in range(matrix.shape[1])]
    lines = [" ".join(header)]
    for idx, row in enumerate(matrix):
        values = " ".join(f"{value:>5}" for value in row)
        lines.append(f"{idx:>9} {values}")
    return "\n".join(lines)


def format_class_metrics(metrics: dict) -> str:
    lines = ["class       precision  recall   f1-score"]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(
            f"{name:<10} {metrics['precision_per_class'][idx] * 100:>8.2f}%"
            f" {metrics['recall_per_class'][idx] * 100:>7.2f}%"
            f" {metrics['f1_per_class'][idx] * 100:>9.2f}%"
        )
    return "\n".join(lines)


def make_folds(num_samples: int, num_folds: int) -> list[np.ndarray]:
    indices = np.arange(num_samples)
    return list(np.array_split(indices, num_folds))


def cross_validate_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_folds: int,
) -> dict:
    folds = make_folds(x_train.shape[0], num_folds)
    fold_accuracies = []

    for fold_idx in range(num_folds):
        val_indices = folds[fold_idx]
        train_indices = np.concatenate(
            [folds[idx] for idx in range(num_folds) if idx != fold_idx],
            axis=0,
        )

        fold_train = x_train[train_indices] / 255.0
        fold_val = x_train[val_indices] / 255.0
        fold_train, fold_val = normalize_with_train_stats(fold_train, fold_val)
        fold_train = add_bias(fold_train)
        fold_val = add_bias(fold_val)

        fold_y_train = y_train[train_indices]
        fold_y_val = y_train[val_indices]
        encoded_fold_y_train = one_hot(fold_y_train, num_classes=len(CLASS_NAMES))

        weights = train_linear_regression(fold_train, encoded_fold_y_train)
        preds = predict(fold_val, weights)
        fold_accuracies.append(accuracy(fold_y_val, preds))

    return {
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": float(np.mean(fold_accuracies)),
    }


def evaluate_on_test(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    y_train_encoded = one_hot(y_train, num_classes=len(CLASS_NAMES))
    weights = train_linear_regression(x_train, y_train_encoded)
    preds = predict(x_test, weights)
    matrix = confusion_matrix(y_test, preds, num_classes=len(CLASS_NAMES))
    metrics = classification_metrics(matrix)
    return {
        "predictions": preds,
        "accuracy": accuracy(y_test, preds),
        "confusion_matrix": matrix,
        "metrics": metrics,
    }


def save_cv_plot(cv_result: dict, output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save the cross-validation plot.") from exc

    fold_indices = np.arange(1, len(cv_result["fold_accuracies"]) + 1)
    accuracies = [score * 100.0 for score in cv_result["fold_accuracies"]]
    mean_accuracy = cv_result["mean_accuracy"] * 100.0

    plt.figure(figsize=(8, 5))
    plt.plot(fold_indices, accuracies, marker="o", linewidth=2, label="Fold accuracy")
    plt.axhline(mean_accuracy, color="red", linestyle="--", label=f"Mean = {mean_accuracy:.2f}%")
    plt.xticks(fold_indices)
    plt.xlabel("Fold")
    plt.ylabel("Validation accuracy (%)")
    plt.title("5-Fold Cross-Validation for Pure Linear Regression")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_confusion_matrix_plot(matrix: np.ndarray, output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save the confusion matrix plot.") from exc

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix on CIFAR-10 Test Set")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_class_metrics_plot(metrics: dict, output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save the class metrics plot.") from exc

    positions = np.arange(len(CLASS_NAMES))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(positions - width, metrics["precision_per_class"] * 100.0, width=width, label="Precision")
    plt.bar(positions, metrics["recall_per_class"] * 100.0, width=width, label="Recall")
    plt.bar(positions + width, metrics["f1_per_class"] * 100.0, width=width, label="F1-score")
    plt.xticks(positions, CLASS_NAMES, rotation=45, ha="right")
    plt.ylabel("Score (%)")
    plt.xlabel("Class")
    plt.title("Per-Class Precision, Recall, and F1-Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("cifar-10-batches-py"),
        help="Path to the extracted CIFAR-10 python batches.",
    )
    parser.add_argument("--num-train", type=int, default=5000, help="Training subset size.")
    parser.add_argument("--num-test", type=int, default=1000, help="Test subset size.")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("linear_regression_5fold_cv.png"),
        help="Path to save the cross-validation plot.",
    )
    parser.add_argument(
        "--confusion-plot-path",
        type=Path,
        default=Path("linear_regression_confusion_matrix.png"),
        help="Path to save the confusion matrix heatmap.",
    )
    parser.add_argument(
        "--metrics-plot-path",
        type=Path,
        default=Path("linear_regression_class_metrics.png"),
        help="Path to save the per-class metrics bar chart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_x_train, raw_y_train, raw_x_test, raw_y_test = load_cifar10(args.data_dir)

    raw_x_train = raw_x_train[: args.num_train]
    raw_y_train = raw_y_train[: args.num_train]

    cv_result = cross_validate_model(
        x_train=raw_x_train,
        y_train=raw_y_train,
        num_folds=args.num_folds,
    )

    prepared_x_train, prepared_x_test = prepare_train_test_features(
        raw_x_train,
        raw_x_test,
        args.num_train,
        args.num_test,
    )
    y_test = raw_y_test[: args.num_test]

    test_result = evaluate_on_test(
        x_train=prepared_x_train,
        y_train=raw_y_train,
        x_test=prepared_x_test,
        y_test=y_test,
    )

    save_cv_plot(cv_result, args.plot_path)
    save_confusion_matrix_plot(test_result["confusion_matrix"], args.confusion_plot_path)
    save_class_metrics_plot(test_result["metrics"], args.metrics_plot_path)

    print("CIFAR-10 Pure Linear Regression with 5-Fold Cross-Validation")
    print(f"train samples: {args.num_train}")
    print(f"test samples:  {args.num_test}")
    print(f"folds:         {args.num_folds}")
    print(f"feature size:  {prepared_x_train.shape[1] - 1} + bias")
    print()
    print("Cross-validation results")
    for fold_idx, score in enumerate(cv_result["fold_accuracies"], start=1):
        print(f"fold {fold_idx}: validation accuracy={score * 100:.2f}%")

    print()
    print(f"Mean 5-fold validation accuracy: {cv_result['mean_accuracy'] * 100:.2f}%")
    print(f"Test-set accuracy: {test_result['accuracy'] * 100:.2f}%")
    print(f"Macro precision: {test_result['metrics']['macro_precision'] * 100:.2f}%")
    print(f"Macro recall:    {test_result['metrics']['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-score:  {test_result['metrics']['macro_f1'] * 100:.2f}%")
    print(f"Cross-validation plot saved to: {args.plot_path}")
    print(f"Confusion matrix plot saved to: {args.confusion_plot_path}")
    print(f"Class metrics plot saved to:    {args.metrics_plot_path}")
    print()
    print("Confusion matrix for test set")
    print(format_confusion_matrix(test_result["confusion_matrix"]))
    print()
    print("Per-class precision, recall, and F1-score")
    print(format_class_metrics(test_result["metrics"]))
    print()
    print("Class index mapping")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"{idx}: {name}")


if __name__ == "__main__":
    main()
