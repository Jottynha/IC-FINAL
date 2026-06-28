"""Gera diagnosticos agregados a partir dos resultados salvos por seed."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "diabetes_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
DETAILS_PATH = OUTPUT_DIR / "stability_results_by_seed.csv"
MODEL_DIR_NAMES = {
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Rede Neural": "rede_neural",
}
CLASS_LABELS = {
    0: "N\u00e3o readmitido <30",
    1: "Readmitido <30",
}


def load_test_supports(seeds):
    y = pd.read_csv(DATASET_PATH, usecols=["readmitted"])["readmitted"].astype(int)
    supports = {}
    for seed in seeds:
        _, y_test = train_test_split(
            y,
            test_size=0.2,
            random_state=int(seed),
            stratify=y,
        )
        counts = y_test.value_counts()
        supports[int(seed)] = {
            0: int(counts.get(0, 0)),
            1: int(counts.get(1, 0)),
        }
    return supports


def reconstruct_confusion_matrices(details):
    supports = load_test_supports(details["seed"].unique())
    rows = []

    for _, result in details.iterrows():
        seed = int(result["seed"])
        negative_support = supports[seed][0]
        positive_support = supports[seed][1]
        total = negative_support + positive_support

        true_positive = int(round(result["recall"] * positive_support))
        true_negative = int(round(result["accuracy"] * total)) - true_positive
        false_negative = positive_support - true_positive
        false_positive = negative_support - true_negative

        counts = [true_negative, false_positive, false_negative, true_positive]
        if any(value < 0 for value in counts):
            raise ValueError(
                f"Matriz invalida para {result['Modelo']} na seed {seed}: {counts}"
            )

        calculated_precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        if not np.isclose(calculated_precision, result["precision"], atol=1e-12):
            raise ValueError(
                f"Precisao reconstruida diverge para {result['Modelo']} "
                f"na seed {seed}"
            )

        rows.append({
            "Modelo": result["Modelo"],
            "seed": seed,
            "TN": true_negative,
            "FP": false_positive,
            "FN": false_negative,
            "TP": true_positive,
            "TN_normalizado": true_negative / negative_support,
            "FP_normalizado": false_positive / negative_support,
            "FN_normalizado": false_negative / positive_support,
            "TP_normalizado": true_positive / positive_support,
        })

    return pd.DataFrame(rows)


def calculate_classification_reports(confusions):
    rows = []
    for _, matrix in confusions.iterrows():
        tn, fp = int(matrix["TN"]), int(matrix["FP"])
        fn, tp = int(matrix["FN"]), int(matrix["TP"])

        class_values = {
            0: {
                "precision": tn / (tn + fn) if tn + fn else 0.0,
                "recall": tn / (tn + fp) if tn + fp else 0.0,
                "support": tn + fp,
            },
            1: {
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "support": tp + fn,
            },
        }

        for class_id, values in class_values.items():
            precision = values["precision"]
            recall = values["recall"]
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            rows.append({
                "Modelo": matrix["Modelo"],
                "seed": int(matrix["seed"]),
                "Classe": CLASS_LABELS[class_id],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(values["support"]),
            })

    return pd.DataFrame(rows)


def summarize_classification_reports(reports):
    rows = []
    for (model_name, class_name), group in reports.groupby(
        ["Modelo", "Classe"],
        sort=False,
    ):
        rows.append({
            "Modelo": model_name,
            "Classe": class_name,
            "precision_media": group["precision"].mean(),
            "precision_desvio": group["precision"].std(ddof=1),
            "recall_media": group["recall"].mean(),
            "recall_desvio": group["recall"].std(ddof=1),
            "f1_media": group["f1"].mean(),
            "f1_desvio": group["f1"].std(ddof=1),
            "support_media": group["support"].mean(),
        })
    return pd.DataFrame(rows)


def format_classification_report_for_article(summary):
    rows = []
    for _, result in summary.iterrows():
        rows.append({
            "Modelo": result["Modelo"],
            "Classe": result["Classe"],
            "Precis\u00e3o": (
                f"{result['precision_media']:.4f} \u00b1 "
                f"{result['precision_desvio']:.4f}"
            ),
            "Revoca\u00e7\u00e3o": (
                f"{result['recall_media']:.4f} \u00b1 "
                f"{result['recall_desvio']:.4f}"
            ),
            "F1-score": (
                f"{result['f1_media']:.4f} \u00b1 "
                f"{result['f1_desvio']:.4f}"
            ),
            "Suporte": f"{result['support_media']:.0f}",
        })
    return pd.DataFrame(rows)


def summarize_confusion_matrices(confusions):
    metrics = [
        "TN",
        "FP",
        "FN",
        "TP",
        "TN_normalizado",
        "FP_normalizado",
        "FN_normalizado",
        "TP_normalizado",
    ]
    rows = []
    for model_name, group in confusions.groupby("Modelo", sort=False):
        row = {"Modelo": model_name}
        for metric in metrics:
            row[f"{metric}_media"] = group[metric].mean()
            row[f"{metric}_desvio"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_mean_confusion_matrices(summary):
    sns.set_style("white")
    for _, result in summary.iterrows():
        means = np.array([
            [result["TN_normalizado_media"], result["FP_normalizado_media"]],
            [result["FN_normalizado_media"], result["TP_normalizado_media"]],
        ])

        annotations = np.empty_like(means, dtype=object)
        for row_index in range(2):
            for column_index in range(2):
                annotations[row_index, column_index] = (
                    f"{means[row_index, column_index] * 100:.1f}%"
                )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            means * 100,
            annot=annotations,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            cbar_kws={"label": "Porcentagem m\u00e9dia (%)"},
            ax=ax,
        )
        labels = [CLASS_LABELS[0], CLASS_LABELS[1]]
        ax.set_xticklabels(labels, rotation=0)
        ax.set_yticklabels(labels, rotation=0)
        ax.set_xlabel("Classe predita")
        ax.set_ylabel("Classe verdadeira")
        ax.set_title(
            f"{result['Modelo']} - matriz de confus\u00e3o normalizada m\u00e9dia"
        )
        fig.tight_layout()

        model_dir = OUTPUT_DIR / "algoritmos" / MODEL_DIR_NAMES[result["Modelo"]]
        model_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            model_dir / "matriz_confusao_media_normalizada.png",
            dpi=300,
        )
        plt.close(fig)


def generate_stability_diagnostics(details=None):
    if details is None:
        details = pd.read_csv(DETAILS_PATH)

    confusions = reconstruct_confusion_matrices(details)
    confusions.to_csv(
        OUTPUT_DIR / "stability_confusion_matrices_by_seed.csv",
        index=False,
    )

    confusion_summary = summarize_confusion_matrices(confusions)
    confusion_summary.to_csv(
        OUTPUT_DIR / "stability_confusion_matrices_summary.csv",
        index=False,
    )

    reports = calculate_classification_reports(confusions)
    reports.to_csv(
        OUTPUT_DIR / "stability_classification_report_by_seed.csv",
        index=False,
    )

    report_summary = summarize_classification_reports(reports)
    report_summary.to_csv(
        OUTPUT_DIR / "stability_classification_report_summary.csv",
        index=False,
    )

    article_table = format_classification_report_for_article(report_summary)
    article_table.to_csv(
        OUTPUT_DIR / "stability_classification_report_article_table.csv",
        index=False,
    )

    plot_mean_confusion_matrices(confusion_summary)
    return article_table


def main():
    article_table = generate_stability_diagnostics()
    print("\nRelatorio de classificacao por classe:")
    print(article_table.to_string(index=False))


if __name__ == "__main__":
    main()
