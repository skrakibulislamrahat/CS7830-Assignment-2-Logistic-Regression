"""CS 7830 Assignment 2.

I used NumPy, pandas, and Matplotlib. Flu.csv is read from this folder and the
tables and graphs are saved inside the results folder.
"""

from pathlib import Path
from math import erfc, sqrt
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Flu.csv"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

RANDOM_SEED = 42
TEST_FRACTION = 0.20
THRESHOLD = 0.50

NUMERIC_EDA_FEATURES = [
    "Vaccin",
    "HndWshQual",
    "HndWshFreq",
    "SociDist",
    "NoFaceContact",
    "RespEttiqu",
    "PersnDist",
    "HandSanit",
]

MODEL_FEATURES = [
    "Vaccin",
    "HndWshQual",
    "HndWshFreq",
    "SociDist",
    "NoFaceContact",
    "RespEttiqu",
    "PersnDist",
    "HandSanit",
    "Risk",
    "Inefficacy",
    "KnowlTrans",
    "KnowlMgmt",
]


def ensure_output_folders():
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)


def sigmoid(values):
    """Sigmoid function used by every logistic regression model."""
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class Standardizer:
    """Store the training mean and standard deviation for scaling."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        self.mean_ = np.mean(x, axis=0)
        self.std_ = np.std(x, axis=0)
        self.std_[self.std_ == 0.0] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def fit_transform(self, x):
        return self.fit(x).transform(x)


class LogisticRegressionScratch:
    """Logistic regression trained with batch gradient descent."""

    def __init__(
        self,
        learning_rate=0.05,
        max_iterations=50000,
        tolerance=1e-10,
        l2_strength=0.0,
        cost_function="cross_entropy",
    ):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.l2_strength = l2_strength
        self.cost_function = cost_function
        self.theta = None
        self.cost_history = []
        self.iterations_run = 0

    @staticmethod
    def add_intercept(x):
        return np.column_stack([np.ones(x.shape[0]), x])

    def fit(self, x, y, initial_theta=None):
        x_bias = self.add_intercept(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        sample_count = len(y)

        if initial_theta is None:
            theta = np.zeros(x_bias.shape[1], dtype=float)
        else:
            theta = np.asarray(initial_theta, dtype=float).copy()
            if len(theta) != x_bias.shape[1]:
                raise ValueError("initial_theta has the wrong length")

        self.cost_history = []

        for iteration in range(self.max_iterations):
            probabilities = sigmoid(x_bias @ theta)

            if self.cost_function == "cross_entropy":
                safe_probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
                data_cost = -np.mean(
                    y * np.log(safe_probabilities)
                    + (1.0 - y) * np.log(1.0 - safe_probabilities)
                )
                gradient = x_bias.T @ (probabilities - y) / sample_count
            elif self.cost_function == "mse":
                error = probabilities - y
                data_cost = np.mean(error**2) / 2.0
                # Chain rule: derivative of sigmoid is p * (1 - p).
                gradient = (
                    x_bias.T @ (error * probabilities * (1.0 - probabilities))
                    / sample_count
                )
            else:
                raise ValueError("cost_function must be 'cross_entropy' or 'mse'")

            penalty = (
                self.l2_strength
                * np.sum(theta[1:] ** 2)
                / (2.0 * sample_count)
            )
            gradient[1:] += self.l2_strength * theta[1:] / sample_count
            total_cost = data_cost + penalty
            self.cost_history.append(float(total_cost))

            updated_theta = theta - self.learning_rate * gradient
            if np.max(np.abs(updated_theta - theta)) < self.tolerance:
                theta = updated_theta
                self.iterations_run = iteration + 1
                break
            theta = updated_theta
        else:
            self.iterations_run = self.max_iterations

        self.theta = theta
        return self

    def predict_proba(self, x):
        if self.theta is None:
            raise RuntimeError("Fit the model before predicting")
        x_bias = self.add_intercept(np.asarray(x, dtype=float))
        return sigmoid(x_bias @ self.theta)

    def predict(self, x, threshold=THRESHOLD):
        return (self.predict_proba(x) >= threshold).astype(int)


def stratified_train_test_indices(y, test_fraction=TEST_FRACTION, seed=RANDOM_SEED):
    """Split each class separately so both sets keep a similar class ratio."""
    y = np.asarray(y)
    generator = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for class_value in np.sort(np.unique(y)):
        class_indices = np.where(y == class_value)[0].copy()
        generator.shuffle(class_indices)
        test_count = int(round(len(class_indices) * test_fraction))
        test_indices.extend(class_indices[:test_count])
        train_indices.extend(class_indices[test_count:])

    train_indices = np.array(train_indices, dtype=int)
    test_indices = np.array(test_indices, dtype=int)
    generator.shuffle(train_indices)
    generator.shuffle(test_indices)
    return train_indices, test_indices


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    return true_negative, false_positive, false_negative, true_positive


def roc_curve_from_scratch(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(-probabilities)
    sorted_y = y_true[order]
    positives = np.sum(sorted_y == 1)
    negatives = np.sum(sorted_y == 0)
    true_positive = np.cumsum(sorted_y == 1)
    false_positive = np.cumsum(sorted_y == 0)
    tpr = np.concatenate([[0.0], true_positive / positives, [1.0]])
    fpr = np.concatenate([[0.0], false_positive / negatives, [1.0]])
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def classification_metrics(y_true, probabilities, threshold=THRESHOLD):
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_counts(y_true, predictions)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2.0
    _, _, auc = roc_curve_from_scratch(y_true, probabilities)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    log_loss = -np.mean(
        y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "BalancedAccuracy": balanced_accuracy,
        "AUC": auc,
        "LogLoss": log_loss,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def fit_and_evaluate(
    train_x,
    train_y,
    test_x,
    test_y,
    learning_rate=0.05,
    l2_strength=0.0,
    scale_features=False,
    cost_function="cross_entropy",
    initial_theta=None,
    max_iterations=50000,
):
    standardizer = None
    if scale_features:
        standardizer = Standardizer()
        train_used = standardizer.fit_transform(train_x)
        test_used = standardizer.transform(test_x)
    else:
        train_used = np.asarray(train_x, dtype=float)
        test_used = np.asarray(test_x, dtype=float)

    model = LogisticRegressionScratch(
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        tolerance=1e-10,
        l2_strength=l2_strength,
        cost_function=cost_function,
    ).fit(train_used, train_y, initial_theta=initial_theta)

    probabilities = model.predict_proba(test_used)
    metrics = classification_metrics(test_y, probabilities)
    return model, standardizer, probabilities, metrics


def normal_two_sided_pvalue(z_value):
    return erfc(abs(float(z_value)) / sqrt(2.0))


def coefficient_pvalues(x, model):
    """Calculate coefficient z-values and Wald p-values."""
    x_bias = model.add_intercept(np.asarray(x, dtype=float))
    probabilities = model.predict_proba(x)
    weights = probabilities * (1.0 - probabilities)
    information = (x_bias.T * weights) @ x_bias
    covariance = np.linalg.pinv(information)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 1e-30))
    z_values = model.theta / standard_errors
    p_values = np.array([normal_two_sided_pvalue(value) for value in z_values])
    return standard_errors, z_values, p_values


def forward_selection(x_frame, y, max_steps=5, entry_threshold=0.05):
    """Add the lowest-p-value feature at each forward-selection step."""
    selected = []
    remaining = list(x_frame.columns)
    selection_rows = []

    for step in range(1, max_steps + 1):
        candidates = []
        for candidate in remaining:
            candidate_features = selected + [candidate]
            x_candidate = x_frame[candidate_features].to_numpy(dtype=float)
            model = LogisticRegressionScratch(
                learning_rate=0.04,
                max_iterations=50000,
                tolerance=1e-10,
            ).fit(x_candidate, y)
            _, z_values, p_values = coefficient_pvalues(x_candidate, model)
            candidates.append(
                {
                    "Step": step,
                    "Candidate": candidate,
                    "PValue": float(p_values[-1]),
                    "ZValue": float(z_values[-1]),
                    "TrainingCost": float(model.cost_history[-1]),
                }
            )

        candidates.sort(key=lambda row: (row["PValue"], row["TrainingCost"]))
        best = candidates[0]
        accepted = best["PValue"] < entry_threshold
        best["Accepted"] = accepted
        best["SelectedBeforeStep"] = ", ".join(selected) if selected else "None"

        if accepted:
            selected.append(best["Candidate"])
            remaining.remove(best["Candidate"])
            best["SelectedAfterStep"] = ", ".join(selected)
            selection_rows.append(best)
        else:
            best["SelectedAfterStep"] = ", ".join(selected)
            selection_rows.append(best)
            break

    return selected, pd.DataFrame(selection_rows)


def plot_confusion_matrix(ax, metrics, title):
    matrix = np.array(
        [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]]
    )
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    for row in range(2):
        for column in range(2):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title(title)
    return image


def exploratory_analysis(data):
    numeric_summary = data[NUMERIC_EDA_FEATURES].describe().T
    numeric_summary.to_csv(TABLES_DIR / "q1_numeric_summary.csv", float_format="%.6f")

    converted = data.copy()
    converted["FemaleCategory"] = converted["Female"].map({0.0: "No", 1.0: "Yes"})
    converted["SickCategory"] = converted["Sick"].map(
        {0.0: "Not sick", 1.0: "Cold-like symptoms", 2.0: "Flu-like symptoms"}
    )
    converted["FluCategory"] = converted["Flu"].map(
        {0.0: "No flu-like illness", 1.0: "Flu-like illness"}
    )
    category_columns = ["FemaleCategory", "SickCategory", "FluCategory"]
    categorical_summary = converted[category_columns].describe().T
    categorical_summary.to_csv(TABLES_DIR / "q1_categorical_summary.csv")

    correlations = data[MODEL_FEATURES].corr()
    correlations.to_csv(TABLES_DIR / "q1_correlations.csv", float_format="%.6f")

    figure, axis = plt.subplots(figsize=(8.3, 7.0))
    image = axis.imshow(correlations.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(MODEL_FEATURES)), MODEL_FEATURES, rotation=55, ha="right")
    axis.set_yticks(range(len(MODEL_FEATURES)), MODEL_FEATURES)
    axis.set_title("Correlation matrix for the 12 model features")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "q1_correlation_matrix.png", dpi=220)
    plt.close(figure)

    return numeric_summary, categorical_summary, correlations


def question_two(data):
    usable = data[["Risk", "Flu"]].dropna().reset_index(drop=True)
    x = usable[["Risk"]].to_numpy(dtype=float)
    y = usable["Flu"].to_numpy(dtype=int)
    train_indices, test_indices = stratified_train_test_indices(y)

    model, _, probabilities, metrics = fit_and_evaluate(
        x[train_indices],
        y[train_indices],
        x[test_indices],
        y[test_indices],
        learning_rate=0.08,
    )

    x_line = np.linspace(x[:, 0].min(), x[:, 0].max(), 300).reshape(-1, 1)
    probability_line = model.predict_proba(x_line)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].scatter(x[test_indices, 0], y[test_indices], alpha=0.65, s=30, label="Test observations")
    axes[0].plot(x_line[:, 0], probability_line, color="#b22222", linewidth=2.2, label="Predicted probability")
    axes[0].axhline(THRESHOLD, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Risk")
    axes[0].set_ylabel("Flu / predicted probability")
    axes[0].set_title("One-variable logistic regression")
    axes[0].legend(fontsize=8)
    plot_confusion_matrix(axes[1], metrics, "Test confusion matrix")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "q2_risk_model.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    axis.plot(model.cost_history, color="#1f5f8b")
    axis.set_xlabel("Gradient descent iteration")
    axis.set_ylabel("Cross-entropy cost")
    axis.set_title("Q2 cost decreases during training")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "q2_cost_history.png", dpi=220)
    plt.close(figure)

    return {
        "model": model,
        "metrics": metrics,
        "rows": len(usable),
        "train_rows": len(train_indices),
        "test_rows": len(test_indices),
        "train_positive": int(y[train_indices].sum()),
        "test_positive": int(y[test_indices].sum()),
    }


def question_three(data):
    usable = data[MODEL_FEATURES + ["Flu"]].dropna().reset_index(drop=True)
    x_frame = usable[MODEL_FEATURES]
    y = usable["Flu"].to_numpy(dtype=int)
    train_indices, test_indices = stratified_train_test_indices(y)

    train_frame = x_frame.iloc[train_indices].reset_index(drop=True)
    test_frame = x_frame.iloc[test_indices].reset_index(drop=True)
    train_y = y[train_indices]
    test_y = y[test_indices]

    full_model, _, full_probabilities, full_metrics = fit_and_evaluate(
        train_frame.to_numpy(dtype=float),
        train_y,
        test_frame.to_numpy(dtype=float),
        test_y,
        learning_rate=0.03,
    )

    selected_features, selection_table = forward_selection(train_frame, train_y)
    if not selected_features:
        raise RuntimeError("Forward selection did not select any feature")
    selection_table.to_csv(TABLES_DIR / "q3_forward_selection.csv", index=False, float_format="%.8f")

    selected_model, _, selected_probabilities, selected_metrics = fit_and_evaluate(
        train_frame[selected_features].to_numpy(dtype=float),
        train_y,
        test_frame[selected_features].to_numpy(dtype=float),
        test_y,
        learning_rate=0.04,
    )

    comparison = pd.DataFrame(
        [
            {"Model": "All 12 features", **full_metrics},
            {"Model": "Forward-selected features", **selected_metrics},
        ]
    )
    comparison.to_csv(TABLES_DIR / "q3_model_comparison.csv", index=False, float_format="%.6f")

    coefficient_rows = []
    for name, value in zip(["Intercept"] + MODEL_FEATURES, full_model.theta):
        coefficient_rows.append({"Model": "All 12 features", "Feature": name, "Theta": value})
    for name, value in zip(["Intercept"] + selected_features, selected_model.theta):
        coefficient_rows.append({"Model": "Forward-selected features", "Feature": name, "Theta": value})
    pd.DataFrame(coefficient_rows).to_csv(
        TABLES_DIR / "q3_coefficients.csv", index=False, float_format="%.8f"
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    plot_confusion_matrix(axes[0], full_metrics, "All 12 features")
    plot_confusion_matrix(axes[1], selected_metrics, "Forward-selected features")
    figure.suptitle("Q3 held-out test confusion matrices")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "q3_confusion_comparison.png", dpi=220)
    plt.close(figure)

    metric_names = ["Accuracy", "Precision", "Recall", "F1", "BalancedAccuracy", "AUC"]
    positions = np.arange(len(metric_names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.0, 4.1))
    axis.bar(positions - width / 2, [full_metrics[name] for name in metric_names], width, label="All 12")
    axis.bar(positions + width / 2, [selected_metrics[name] for name in metric_names], width, label="Selected")
    axis.set_xticks(positions, metric_names, rotation=25, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Q3 test performance")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "q3_metric_comparison.png", dpi=220)
    plt.close(figure)

    return {
        "usable": usable,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_frame": train_frame,
        "test_frame": test_frame,
        "train_y": train_y,
        "test_y": test_y,
        "selected_features": selected_features,
        "selection_table": selection_table,
        "full_model": full_model,
        "full_metrics": full_metrics,
        "selected_model": selected_model,
        "selected_metrics": selected_metrics,
    }


def choose_regularization_strength(train_x, train_y, seed=314):
    """Choose lambda on a validation part of the training data."""
    subtrain_indices, validation_indices = stratified_train_test_indices(
        train_y, test_fraction=0.25, seed=seed
    )
    lambda_values = [0.01, 0.1, 1.0, 10.0]
    rows = []
    for value in lambda_values:
        model, _, probabilities, metrics = fit_and_evaluate(
            train_x[subtrain_indices],
            train_y[subtrain_indices],
            train_x[validation_indices],
            train_y[validation_indices],
            learning_rate=0.04,
            l2_strength=value,
        )
        rows.append(
            {
                "Lambda": value,
                "ValidationLogLoss": metrics["LogLoss"],
                "ValidationAccuracy": metrics["Accuracy"],
                "Iterations": model.iterations_run,
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["ValidationLogLoss", "Lambda"], ascending=[True, True]
    )
    return float(table.iloc[0]["Lambda"]), table


def question_four(q3_result):
    selected_features = q3_result["selected_features"]
    train_frame = q3_result["train_frame"]
    test_frame = q3_result["test_frame"]
    train_y = q3_result["train_y"]
    test_y = q3_result["test_y"]
    train_x = train_frame[selected_features].to_numpy(dtype=float)
    test_x = test_frame[selected_features].to_numpy(dtype=float)

    chosen_lambda, lambda_table = choose_regularization_strength(train_x, train_y)
    lambda_table.to_csv(TABLES_DIR / "q4_lambda_selection.csv", index=False, float_format="%.8f")

    configurations = [
        ("Baseline", False, 0.0, 0.04),
        (f"L2 regularization (lambda={chosen_lambda:g})", False, chosen_lambda, 0.04),
        ("Feature scaling", True, 0.0, 0.10),
        (f"Scaling + L2 (lambda={chosen_lambda:g})", True, chosen_lambda, 0.10),
    ]

    rows = []
    fitted = {}
    for label, scaled, l2_value, learning_rate in configurations:
        model, standardizer, probabilities, metrics = fit_and_evaluate(
            train_x,
            train_y,
            test_x,
            test_y,
            learning_rate=learning_rate,
            l2_strength=l2_value,
            scale_features=scaled,
        )
        rows.append(
            {
                "Model": label,
                "Scaled": scaled,
                "Lambda": l2_value,
                "Iterations": model.iterations_run,
                **metrics,
            }
        )
        fitted[label] = (model, standardizer, probabilities, metrics)

    experiment_table = pd.DataFrame(rows)
    experiment_table.to_csv(TABLES_DIR / "q4_regularization_scaling.csv", index=False, float_format="%.6f")

    # Scaling reaches the same solution much faster, so it is retained for Q4b.
    best_label = "Feature scaling"
    best_model, best_standardizer, _, best_metrics = fitted[best_label]
    scaled_train_x = best_standardizer.transform(train_x)
    scaled_test_x = best_standardizer.transform(test_x)

    mse_runs = []
    mse_models = []
    for seed in [7, 42, 99, 2025, 1337]:
        generator = np.random.default_rng(seed)
        initial_theta = generator.normal(0.0, 0.10, scaled_train_x.shape[1] + 1)
        model, _, probabilities, metrics = fit_and_evaluate(
            scaled_train_x,
            train_y,
            scaled_test_x,
            test_y,
            learning_rate=0.20,
            cost_function="mse",
            initial_theta=initial_theta,
            max_iterations=100000,
        )
        mse_runs.append(
            {
                "Seed": seed,
                "FinalMSECost": model.cost_history[-1],
                "Iterations": model.iterations_run,
                **metrics,
            }
        )
        mse_models.append(model)

    mse_table = pd.DataFrame(mse_runs).sort_values(["FinalMSECost", "Seed"])
    mse_table.to_csv(TABLES_DIR / "q4_mse_restarts.csv", index=False, float_format="%.8f")
    chosen_mse_model = mse_models[int(np.argmin([model.cost_history[-1] for model in mse_models]))]
    mse_probabilities = chosen_mse_model.predict_proba(scaled_test_x)
    mse_metrics = classification_metrics(test_y, mse_probabilities)

    cost_comparison = pd.DataFrame(
        [
            {"Model": "Cross-entropy with scaling", **best_metrics},
            {"Model": "MSE with scaling", **mse_metrics},
        ]
    )
    cost_comparison.to_csv(TABLES_DIR / "q4_cost_comparison.csv", index=False, float_format="%.6f")

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    plot_confusion_matrix(axes[0], best_metrics, "Cross-entropy")
    plot_confusion_matrix(axes[1], mse_metrics, "Mean squared error")
    figure.suptitle("Q4b held-out test confusion matrices")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "q4_cost_confusion_comparison.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))
    axes[0].plot(best_model.cost_history, color="#1f5f8b")
    axes[0].set_title("Cross-entropy cost")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cost")
    axes[0].grid(alpha=0.25)
    for model, seed in zip(mse_models, [7, 42, 99, 2025, 1337]):
        axes[1].plot(model.cost_history, linewidth=1.1, label=f"Seed {seed}")
    axes[1].set_title("MSE cost from five initializations")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Cost")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "q4_cost_histories.png", dpi=220)
    plt.close(figure)

    return {
        "chosen_lambda": chosen_lambda,
        "experiment_table": experiment_table,
        "best_label": best_label,
        "best_model": best_model,
        "best_metrics": best_metrics,
        "mse_table": mse_table,
        "mse_model": chosen_mse_model,
        "mse_metrics": mse_metrics,
    }


def serializable_metrics(metrics):
    converted = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, int)):
            converted[key] = int(value)
        else:
            converted[key] = float(value)
    return converted


def main():
    ensure_output_folders()
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.name}. Keep it in the same folder as this script."
        )

    data = pd.read_csv(DATA_PATH)
    numeric_summary, categorical_summary, correlations = exploratory_analysis(data)
    q2_result = question_two(data)
    q3_result = question_three(data)
    q4_result = question_four(q3_result)

    all_metrics = pd.DataFrame(
        [
            {"Model": "Q2 Risk only", **q2_result["metrics"]},
            {"Model": "Q3 all 12 features", **q3_result["full_metrics"]},
            {"Model": "Q3 forward-selected", **q3_result["selected_metrics"]},
            {"Model": "Q4a scaled cross-entropy", **q4_result["best_metrics"]},
            {"Model": "Q4b scaled MSE", **q4_result["mse_metrics"]},
        ]
    )
    all_metrics.to_csv(TABLES_DIR / "all_model_metrics.csv", index=False, float_format="%.6f")

    summary = {
        "dataset_rows": int(len(data)),
        "dataset_columns": int(data.shape[1]),
        "q1_soci_persn_correlation": float(correlations.loc["SociDist", "PersnDist"]),
        "q1_knowledge_correlation": float(correlations.loc["KnowlTrans", "KnowlMgmt"]),
        "q2": {
            "rows": q2_result["rows"],
            "train_rows": q2_result["train_rows"],
            "test_rows": q2_result["test_rows"],
            "theta": q2_result["model"].theta.tolist(),
            "iterations": q2_result["model"].iterations_run,
            "metrics": serializable_metrics(q2_result["metrics"]),
        },
        "q3": {
            "rows": int(len(q3_result["usable"])),
            "train_rows": int(len(q3_result["train_indices"])),
            "test_rows": int(len(q3_result["test_indices"])),
            "selected_features": q3_result["selected_features"],
            "full_theta": q3_result["full_model"].theta.tolist(),
            "selected_theta": q3_result["selected_model"].theta.tolist(),
            "full_metrics": serializable_metrics(q3_result["full_metrics"]),
            "selected_metrics": serializable_metrics(q3_result["selected_metrics"]),
        },
        "q4": {
            "chosen_lambda": q4_result["chosen_lambda"],
            "best_configuration": q4_result["best_label"],
            "cross_entropy_metrics": serializable_metrics(q4_result["best_metrics"]),
            "mse_metrics": serializable_metrics(q4_result["mse_metrics"]),
            "mse_theta": q4_result["mse_model"].theta.tolist(),
        },
    }

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Assignment 2 analysis completed successfully.")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Forward-selected features: {', '.join(q3_result['selected_features'])}")
    print(f"Chosen L2 lambda: {q4_result['chosen_lambda']:g}")


if __name__ == "__main__":
    main()
