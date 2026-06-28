"""Repete os experimentos com diferentes seeds para medir estabilidade."""

from ast import literal_eval
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_training import compute_metrics, find_best_threshold, sanitize_column_names
from stability_postprocessing import generate_stability_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "diabetes_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
BEST_PARAMS_PATH = OUTPUT_DIR / "best_hyperparameters.csv"
CV_FOLDS = 3
SEEDS = [7, 21, 42, 84, 126]


def build_estimator(model_name, params, seed):
    if model_name == "Random Forest":
        estimator = RandomForestClassifier(random_state=seed, n_jobs=1)
    elif model_name == "XGBoost":
        estimator = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    elif model_name == "Rede Neural":
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                activation="relu",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=seed,
            )),
        ])
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    estimator.set_params(**params)
    return estimator


def load_best_params():
    rows = pd.read_csv(BEST_PARAMS_PATH)
    params_column = rows.columns[-1]
    return {
        row["Modelo"]: literal_eval(row[params_column])
        for _, row in rows.iterrows()
    }

def summarize_results(details):
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "threshold",
    ]
    summary_rows = []
    for model_name, group in details.groupby("Modelo", sort=False):
        row = {"Modelo": model_name}
        for metric in metrics:
            row[f"{metric}_media"] = group[metric].mean()
            row[f"{metric}_desvio"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def format_summary_for_article(summary):
    columns = {
        "accuracy": "Acur\u00e1cia",
        "precision": "Precis\u00e3o",
        "recall": "Revoca\u00e7\u00e3o",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }
    rows = []
    for _, source in summary.iterrows():
        row = {"Modelo": source["Modelo"]}
        for metric, label in columns.items():
            mean = source[f"{metric}_media"]
            std = source[f"{metric}_desvio"]
            row[label] = f"{mean:.4f} \u00b1 {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)

def plot_stability(summary):
    plot_data = []
    metric_labels = {
        "accuracy": "Acur\u00e1cia",
        "precision": "Precis\u00e3o",
        "recall": "Revoca\u00e7\u00e3o",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
    }
    for _, row in summary.iterrows():
        for metric, label in metric_labels.items():
            plot_data.append({
                "Modelo": row["Modelo"],
                "Metrica": label,
                "Media": row[f"{metric}_media"],
                "Desvio": row[f"{metric}_desvio"],
            })

    data = pd.DataFrame(plot_data)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=data, x="Metrica", y="Media", hue="Modelo", ax=ax)

    patches = ax.patches
    for patch, (_, row) in zip(patches, data.iterrows()):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        deviation = row["Desvio"]
        ax.errorbar(
            x,
            y,
            yerr=deviation,
            color="black",
            capsize=3,
            linewidth=1,
            fmt="none",
        )
        ax.text(
            x,
            y + deviation + 0.012,
            f"{y:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title("Estabilidade dos modelos em diferentes seeds")
    ax.set_xlabel("M\u00e9trica")
    ax.set_ylabel("M\u00e9dia")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "11_estabilidade_seeds.png", dpi=300)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    df = sanitize_column_names(pd.read_csv(DATASET_PATH))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(int)
    best_params = load_best_params()
    rows = []

    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )
        cv = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=seed,
        )

        for model_name, params in best_params.items():
            print(f"Seed {seed} - {model_name}", flush=True)
            estimator = build_estimator(model_name, params, seed)
            parallel_jobs = 1 if model_name == "Rede Neural" else -1
            probabilities_oof = cross_val_predict(
                clone(estimator),
                X_train,
                y_train,
                cv=cv,
                method="predict_proba",
                n_jobs=parallel_jobs,
            )[:, 1]
            threshold_info = find_best_threshold(y_train, probabilities_oof)

            estimator.fit(X_train, y_train)
            test_probabilities = estimator.predict_proba(X_test)[:, 1]
            metrics = compute_metrics(
                y_test,
                test_probabilities,
                threshold_info["threshold"],
            )
            rows.append({
                "Modelo": model_name,
                "seed": seed,
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            })

    details = pd.DataFrame(rows)
    details.to_csv(OUTPUT_DIR / "stability_results_by_seed.csv", index=False)

    summary = summarize_results(details)
    summary.to_csv(OUTPUT_DIR / "stability_results_summary.csv", index=False)

    article_table = format_summary_for_article(summary)
    article_table.to_csv(
        OUTPUT_DIR / "stability_results_article_table.csv",
        index=False,
    )
    plot_stability(summary)
    generate_stability_diagnostics(details)

    print("\nResumo de estabilidade:")
    print(article_table.to_string(index=False))


if __name__ == "__main__":
    main()
