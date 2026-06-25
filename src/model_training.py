"""Testa readmissão em menos de 30 dias como problema binário."""

from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "dataset" / "diabetes_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
CV_FOLDS = 3


def sanitize_column_names(df):
    clean_columns = []
    seen = {}
    for column in df.columns:
        clean = re.sub(r"[\[\]<>]", "", str(column))
        clean = re.sub(r"[^0-9a-zA-Z_]+", "_", clean).strip("_")
        clean = clean or "feature"
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        clean_columns.append(clean)
    df = df.copy()
    df.columns = clean_columns
    return df


def build_searches(positive_weight):
    estimators = {
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "XGBoost": xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        ),
        "Rede Neural": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                activation="relu",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=RANDOM_STATE,
            )),
        ]),
    }
    grids = {
        "Random Forest": {
            "n_estimators": [120, 200],
            "max_depth": [16, 24],
            "min_samples_split": [10],
            "min_samples_leaf": [8],
            "class_weight": ["balanced", "balanced_subsample"],
        },
        "XGBoost": {
            "n_estimators": [120, 200],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.9],
            "colsample_bytree": [0.9],
            "reg_lambda": [0.5, 1.0],
            "scale_pos_weight": [1.0, positive_weight],
        },
        "Rede Neural": {
            "mlp__hidden_layer_sizes": [(100,), (150, 75)],
            "mlp__learning_rate_init": [0.001, 0.01],
            "mlp__alpha": [0.001, 0.01],
            "mlp__batch_size": [64],
        },
    }
    return estimators, grids


def find_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )
    best_index = int(np.argmax(f1_values))
    return {
        "threshold": float(thresholds[best_index]),
        "f1": float(f1_values[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
    }


def compute_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "predictions": predictions,
    }


def plot_model_results(model_name, y_test, probabilities, metrics):
    model_dir = OUTPUT_DIR / "algoritmos" / re.sub(
        r"[^0-9a-zA-Z]+", "_", model_name
    ).strip("_").lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, metrics["predictions"])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Não <30", "<30"],
        yticklabels=["Não <30", "<30"],
        ax=ax,
    )
    ax.set_title(f"Matriz de confusão binária - {model_name}")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    fig.tight_layout()
    fig.savefig(model_dir / "matriz_confusao.png", dpi=300)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#4C78A8")
    ax.axhline(y_test.mean(), color="#E45756", linestyle="--")
    ax.set_title(f"Curva precisão-revocação - {model_name}")
    ax.set_xlabel("Revocação")
    ax.set_ylabel("Precisão")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(model_dir / "curva_precision_recall.png", dpi=300)
    plt.close(fig)


def create_comparison_plot(results):
    plot_data = pd.DataFrame([
        {
            "Modelo": model_name,
            "Precisão": values["precision"],
            "Revocação": values["recall"],
            "F1": values["f1"],
            "PR-AUC": values["pr_auc"],
            "ROC-AUC": values["roc_auc"],
        }
        for model_name, values in results.items()
    ]).melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=plot_data,
        x="Métrica",
        y="Valor",
        hue="Modelo",
        ax=ax,
    )
    ax.set_title("Comparação dos modelos com alvo binário")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "10_comparacao_modelos.png", dpi=300)
    plt.close(fig)



def plot_class_distribution(y):
    counts = y.value_counts().sort_index()
    labels = ["Nao readmitido <30", "Readmitido <30"]
    percentages = counts / counts.sum() * 100
    pd.DataFrame({
        "Classe": labels,
        "Codificacao": counts.index,
        "Quantidade": counts.values,
        "Percentual": percentages.values,
    }).to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, counts.values, color=["#4C78A8", "#E45756"])
    ax.set_title("Distribuicao da variavel-alvo binaria")
    ax.set_ylabel("Quantidade de registros")
    for bar, percentage in zip(bars, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "09_distribuicao_classes.png", dpi=300)
    plt.close(fig)


def plot_algorithm_metrics(model_name, metrics):
    model_dir = OUTPUT_DIR / "algoritmos" / re.sub(
        r"[^0-9a-zA-Z]+", "_", model_name
    ).strip("_").lower()
    model_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Acuracia", "Precisao", "Revocacao", "F1", "PR-AUC", "ROC-AUC"]
    values = [
        metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"], metrics["pr_auc"], metrics["roc_auc"],
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, values, color="#4C78A8")
    ax.set_title(f"Metricas binarias no teste - {model_name}")
    ax.set_ylim(0, 1)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(model_dir / "metricas.png", dpi=300)
    plt.close(fig)


def plot_grid_search_results(search_results):
    data = pd.concat(search_results, ignore_index=True).copy()
    data["configuration"] = data.groupby("model").cumcount() + 1
    for model_name in data["model"].unique():
        subset = data[data["model"] == model_name].sort_values("rank_test_score")
        model_dir = OUTPUT_DIR / "algoritmos" / re.sub(
            r"[^0-9a-zA-Z]+", "_", model_name
        ).strip("_").lower()
        model_dir.mkdir(parents=True, exist_ok=True)
        colors = [
            "#54A24B" if rank == 1 else "#4C78A8"
            for rank in subset["rank_test_score"]
        ]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            subset["configuration"].astype(str),
            subset["mean_test_score"],
            color=colors,
        )
        ax.set_title(f"Resultados do GridSearchCV - {model_name}")
        ax.set_xlabel("Configuracao")
        ax.set_ylabel("F1 medio da classe <30")
        ax.set_ylim(0, max(0.01, subset["mean_test_score"].max() * 1.15))
        fig.tight_layout()
        fig.savefig(model_dir / "gridsearchcv_resultados.png", dpi=300)
        plt.close(fig)


def plot_feature_importance(estimators, feature_names):
    for model_name in ("Random Forest", "XGBoost"):
        estimator = estimators[model_name]
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": estimator.feature_importances_,
        }).sort_values("importance", ascending=False).head(15)
        model_dir = OUTPUT_DIR / "algoritmos" / re.sub(
            r"[^0-9a-zA-Z]+", "_", model_name
        ).strip("_").lower()
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(importance["feature"], importance["importance"], color="#4C78A8")
        ax.invert_yaxis()
        ax.set_title(f"Importancia das features - {model_name}")
        fig.tight_layout()
        fig.savefig(model_dir / "importancia_features.png", dpi=300)
        plt.close(fig)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    df = sanitize_column_names(pd.read_csv(DATASET_PATH))
    X = df.iloc[:, :-1]
    multiclass_target = df.iloc[:, -1]
    y = multiclass_target.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    positive_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(
        f"Treino/CV: {len(y_train)}; teste: {len(y_test)}; "
        f"positivos: {y.mean():.2%}; peso positivo: {positive_weight:.3f}"
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    estimators, grids = build_searches(positive_weight)
    results = {}
    trained_estimators = {}
    search_rows = []
    report_blocks = []
    report_rows = []

    for model_name, estimator in estimators.items():
        print(f"\nGridSearchCV: {model_name}", flush=True)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=grids[model_name],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=1,
        )
        search.fit(X_train, y_train)
        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.insert(0, "model", model_name)
        search_rows.append(cv_results)

        best_estimator = search.best_estimator_
        print(
            f"Melhor F1 CV={search.best_score_:.4f}; "
            f"parâmetros={search.best_params_}"
        )
        oof_probabilities = cross_val_predict(
            clone(best_estimator),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        threshold_info = find_best_threshold(y_train, oof_probabilities)
        print(
            f"Limiar OOF={threshold_info['threshold']:.4f}; "
            f"F1 OOF={threshold_info['f1']:.4f}"
        )

        best_estimator.fit(X_train, y_train)
        trained_estimators[model_name] = best_estimator
        test_probabilities = best_estimator.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(
            y_test,
            test_probabilities,
            threshold_info["threshold"],
        )
        metrics["cv_f1_default_threshold"] = search.best_score_
        metrics["oof_f1_tuned_threshold"] = threshold_info["f1"]
        metrics["best_params"] = search.best_params_
        results[model_name] = metrics

        report = classification_report(
            y_test,
            metrics["predictions"],
            target_names=["N\u00e3o readmitido <30", "Readmitido <30"],
            zero_division=0,
        )
        report_dict = classification_report(
            y_test,
            metrics["predictions"],
            target_names=["N\u00e3o readmitido <30", "Readmitido <30"],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report_dict).T
        report_df.insert(0, "Modelo", model_name)
        report_df.insert(1, "Classe", report_df.index)
        report_rows.append(report_df.reset_index(drop=True))
        report_blocks.append(f"{model_name}\n{'-' * len(model_name)}\n{report}")
        plot_model_results(
            model_name, y_test, test_probabilities, metrics
        )
        plot_algorithm_metrics(model_name, metrics)
        filename = re.sub(
            r"[^0-9a-zA-Z]+", "_", model_name
        ).strip("_").lower()
        model_filenames = {
            "random_forest": "random_forest_best.pkl",
            "xgboost": "xgboost_best.pkl",
            "rede_neural": "neural_network_best.pkl",
        }
        joblib.dump(best_estimator, MODELS_DIR / model_filenames[filename])

    joblib.dump(
        {name: values["threshold"] for name, values in results.items()},
        MODELS_DIR / "decision_thresholds.pkl",
    )

    full_search_results = pd.concat(search_rows, ignore_index=True)
    full_search_results.to_csv(
        OUTPUT_DIR / "gridsearchcv_resultados.csv", index=False
    )
    full_search_results.to_csv(
        OUTPUT_DIR / "hyperparameter_search_results.csv", index=False
    )
    plot_class_distribution(y)
    plot_grid_search_results(search_rows)
    plot_feature_importance(trained_estimators, X.columns)
    result_rows = []
    for model_name, values in results.items():
        result_rows.append({
            "Modelo": model_name,
            "Melhor F1 CV com limiar 0.5":
                values["cv_f1_default_threshold"],
            "F1 OOF com limiar ajustado":
                values["oof_f1_tuned_threshold"],
            "Limiar escolhido": values["threshold"],
            "Acurácia teste": values["accuracy"],
            "Precisão <30 teste": values["precision"],
            "Revocação <30 teste": values["recall"],
            "F1 <30 teste": values["f1"],
            "ROC-AUC teste": values["roc_auc"],
            "PR-AUC teste": values["pr_auc"],
            "Melhores hiperparâmetros": str(values["best_params"]),
        })
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(OUTPUT_DIR / "resultados_binarios.csv", index=False)
    results_df[[
        "Modelo", "Melhor F1 CV com limiar 0.5", "Limiar escolhido",
        "Melhores hiperpar\u00e2metros",
    ]].to_csv(OUTPUT_DIR / "best_hyperparameters.csv", index=False)
    pd.concat(report_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "classification_report_por_classe.csv", index=False
    )
    (OUTPUT_DIR / "classification_report_por_classe.txt").write_text(
        "\n\n".join(report_blocks),
        encoding="utf-8",
    )
    create_comparison_plot(results)

    baseline_pr_auc = y_test.mean()
    best_model = max(results, key=lambda name: results[name]["f1"])
    report_text = (
        "EXPERIMENTO COM ALVO BINÁRIO\n"
        + "=" * 72
        + "\n\n"
        + "Classe positiva: readmissão em menos de 30 dias.\n"
        + "Classe negativa: readmissão após 30 dias ou ausência de readmissão.\n"
        + f"Proporção positiva no teste: {baseline_pr_auc:.4f}.\n"
        + "O limiar foi ajustado com previsões out-of-fold do treino.\n\n"
        + results_df.to_string(index=False)
        + f"\n\nMelhor modelo por F1 no teste: {best_model} "
        + f"(F1={results[best_model]['f1']:.4f}).\n"
    )
    (OUTPUT_DIR / "RELATORIO_MODELOS.txt").write_text(
        report_text,
        encoding="utf-8",
    )
    print("\nResultados no teste:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
