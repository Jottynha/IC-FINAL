# Prever readmissão de pacientes em 30 dias
# Algoritmos: Random Forest, XGBoost; Rede Neural (MLP)

from pathlib import Path
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORING_METRIC = "f1_weighted"
CV_FOLDS = 3
CLASS_LABELS = {0: "<30", 1: ">30", 2: "NO"}

HYPERPARAMETER_GRIDS = {
    "Random Forest": {
        "n_estimators": [80, 120],
        "max_depth": [16, 24],
        "min_samples_split": [10],
        "min_samples_leaf": [8],
    },
    "XGBoost": {
        "n_estimators": [80, 120],
        "max_depth": [4, 6],
        "learning_rate": [0.1],
        "subsample": [0.9],
        "colsample_bytree": [0.9],
        "reg_lambda": [0.5],
    },
    "Rede Neural": {
        "mlp__hidden_layer_sizes": [(100,), (150, 75)],
        "mlp__learning_rate_init": [0.001, 0.01],
        "mlp__alpha": [0.001],
        "mlp__batch_size": [64],
    },
}


def sanitize_column_names(df):
    clean_columns = []
    seen = {}

    for col in df.columns:
        clean = re.sub(r"[\[\]<>]", "", str(col))
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


def format_params(params):
    return ", ".join(f"{key}={value}" for key, value in params.items())


def safe_filename(name):
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


class ModelTrainer:
    def __init__(self, csv_path, random_state=42):
        self.random_state = random_state
        self.output_dir = PROJECT_ROOT / 'output'
        self.models_dir = PROJECT_ROOT / 'models'
        self.output_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.search_results = []
        self.best_params = {}
        self.best_cv_scores = {}
        self.df = sanitize_column_names(pd.read_csv(csv_path))
        self._prepare_data()
        print(f"Dataset carregado: {self.X.shape[0]} amostras x {self.X.shape[1]} features")
        print(f"Distribuição de classes:\n{self.y.value_counts().to_string()}")

    def _prepare_data(self):
        self.X = self.df.iloc[:, :-1]
        self.y = self.df.iloc[:, -1]
        # Separar teste final (20%). Os 80% restantes entram no GridSearchCV.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )
    def _algorithm_output_dir(self, model_name):
        path = self.output_dir / 'algoritmos' / safe_filename(model_name)
        path.mkdir(parents=True, exist_ok=True)
        return path
    def _search_hyperparameters(self, model_name, estimator, param_grid, X_train, y_train):
        print(f"\nGridSearchCV ({model_name})")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=self.random_state)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=SCORING_METRIC,
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=1,
        )
        search.fit(X_train, y_train)

        self.best_params[model_name] = search.best_params_
        self.best_cv_scores[model_name] = search.best_score_
        print(f"Melhor configuração ({model_name}): {format_params(search.best_params_)}")
        print(f"Melhor F1 médio em CV ({model_name}): {search.best_score_:.4f}")

        cv_results = pd.DataFrame(search.cv_results_)
        cv_results.insert(0, 'model', model_name)
        self.search_results.append(cv_results)
        return search.best_estimator_

    def train_random_forest(self):
        print("\n" + "="*80)
        print("TREINANDO: RANDOM FOREST")
        print("="*80)
        estimator = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=1,
            class_weight='balanced'
        )
        self.rf_best = self._search_hyperparameters(
            'Random Forest', estimator, HYPERPARAMETER_GRIDS['Random Forest'],
            self.X_train, self.y_train
        )

    def train_xgboost(self):
        print("\n" + "="*80)
        print("TREINANDO: XGBOOST (Gradient Boosting)")
        print("="*80)
        estimator = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=self.random_state,
            n_jobs=1,
            verbosity=0
        )
        self.xgb_best = self._search_hyperparameters(
            'XGBoost', estimator, HYPERPARAMETER_GRIDS['XGBoost'],
            self.X_train, self.y_train
        )

    def train_neural_network(self):
        print("\n" + "="*80)
        print("TREINANDO: REDE NEURAL (MLP)")
        print("="*80)
        estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                activation='relu',
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=0
            )),
        ])
        self.mlp_best = self._search_hyperparameters(
            'Rede Neural', estimator, HYPERPARAMETER_GRIDS['Rede Neural'],
            self.X_train, self.y_train
        )

    def evaluate_all_models_on_test(self):
        print("\n" + "="*80)
        print("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
        print("="*80)
        results = {}
        print("\nRandom Forest")
        y_pred_rf = self.rf_best.predict(self.X_test)
        y_proba_rf = self.rf_best.predict_proba(self.X_test)
        results['Random Forest'] = self._compute_metrics(self.y_test, y_pred_rf, y_proba_rf)
        print("\nXGBoost")
        y_pred_xgb = self.xgb_best.predict(self.X_test)
        y_proba_xgb = self.xgb_best.predict_proba(self.X_test)
        results['XGBoost'] = self._compute_metrics(self.y_test, y_pred_xgb, y_proba_xgb)
        print("\nRede Neural")
        y_pred_mlp = self.mlp_best.predict(self.X_test)
        y_proba_mlp = self.mlp_best.predict_proba(self.X_test)
        results['Rede Neural'] = self._compute_metrics(self.y_test, y_pred_mlp, y_proba_mlp)
        self.results = results

    def _compute_metrics(self, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except ValueError:
            auc = 0.0
        class_report = classification_report(
            y_true,
            y_pred,
            labels=list(CLASS_LABELS.keys()),
            target_names=list(CLASS_LABELS.values()),
            output_dict=True,
            zero_division=0,
        )
        print(f"-> Acurácia={acc:.4f}, Precisão={prec:.4f}, Revocação={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
                'classification_report': class_report, 'y_pred': y_pred, 'y_proba': y_proba}

    def save_hyperparameter_search_results(self):
        print("\n[Salvando] Resultados da busca de hiperparâmetros")
        df = pd.concat(self.search_results, ignore_index=True)
        df['parametros_formatados'] = df['params'].apply(format_params)
        df.to_csv(f'{self.output_dir}/hyperparameter_search_results.csv', index=False)

    def plot_class_distribution(self):
        print("\n[Gerando] Distribuição das classes")
        counts = self.y.value_counts().sort_index()
        labels = [CLASS_LABELS.get(label, str(label)) for label in counts.index]
        percentages = counts / counts.sum() * 100

        distribution = pd.DataFrame({
            'Classe': labels,
            'Codificação': counts.index,
            'Quantidade': counts.values,
            'Percentual': percentages.values,
        })
        distribution.to_csv(f'{self.output_dir}/class_distribution.csv', index=False)

        fig, ax = plt.subplots(figsize=(9, 6))
        bars = ax.bar(labels, counts.values, color=['#4C78A8', '#F58518', '#54A24B'])
        ax.set_title('Distribuição da Variável-Alvo', fontweight='bold')
        ax.set_xlabel('Classe de Readmissão')
        ax.set_ylabel('Quantidade de registros')
        ax.grid(alpha=0.3, axis='y')
        for bar, pct in zip(bars, percentages.values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{int(height):,}\n({pct:.1f}%)'.replace(',', '.'),
                ha='center',
                va='bottom',
                fontsize=10,
            )
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/09_distribuicao_classes.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_best_hyperparameters_table(self):
        print("\n[Salvando] Tabela de melhores hiperparâmetros")
        rows = []
        for model_name in sorted(self.best_params.keys()):
            rows.append({
                'Modelo': model_name,
                'Melhor F1 médio em CV': self.best_cv_scores[model_name],
                'Melhores hiperparâmetros': format_params(self.best_params[model_name]),
            })
        pd.DataFrame(rows).to_csv(f'{self.output_dir}/best_hyperparameters.csv', index=False)

    def plot_grid_search_results(self):
        print("\n[Gerando] Resultados do GridSearchCV por algoritmo")
        df = pd.concat(self.search_results, ignore_index=True).copy()
        df['Configuração'] = df.groupby('model').cumcount() + 1
        df['Configuração'] = df['Configuração'].astype(str)

        for model_name in df['model'].unique():
            subset = df[df['model'] == model_name].sort_values('rank_test_score')
            fig, ax = plt.subplots(figsize=(9, 6))
            colors = ['#54A24B' if rank == 1 else '#4C78A8' for rank in subset['rank_test_score']]
            bars = ax.bar(subset['Configuração'], subset['mean_test_score'], color=colors)
            ax.set_title(f'Resultados do GridSearchCV - {model_name}', fontweight='bold')
            ax.set_xlabel('Configuração')
            ax.set_ylabel('F1-score ponderado médio (CV)')
            ax.set_ylim(0, max(0.01, subset['mean_test_score'].max() * 1.12))
            ax.grid(alpha=0.3, axis='y')
            for bar, score in zip(bars, subset['mean_test_score']):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{score:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )
            plt.tight_layout()
            algorithm_dir = self._algorithm_output_dir(model_name)
            plt.savefig(algorithm_dir / 'gridsearchcv_resultados.png', dpi=300, bbox_inches='tight')
            plt.close()

    def save_classification_reports(self):
        print("\n[Salvando] Classification report por classe")
        rows = []
        text_blocks = []
        for model_name in sorted(self.results.keys()):
            raw_report = self.results[model_name]['classification_report']
            tabular_report = {
                key: value for key, value in raw_report.items()
                if isinstance(value, dict)
            }
            report_df = pd.DataFrame(tabular_report).T
            report_df.insert(0, 'Modelo', model_name)
            report_df.insert(1, 'Classe', report_df.index)
            rows.append(report_df.reset_index(drop=True))
            text_blocks.append(f"{model_name}\n{'-' * len(model_name)}\n")
            text_blocks.append(report_df.to_string(index=False))
            text_blocks.append("\n\n")

        pd.concat(rows, ignore_index=True).to_csv(
            f'{self.output_dir}/classification_report_por_classe.csv', index=False
        )
        with open(f'{self.output_dir}/classification_report_por_classe.txt', 'w', encoding='utf-8') as f:
            f.write(''.join(text_blocks))

    def plot_comparison_results(self):
        print("\n[Gerando] Gráfico de Comparação")
        df = pd.DataFrame({
            'Modelo': list(self.results.keys()),
            'Acurácia': [self.results[m]['accuracy'] for m in self.results.keys()],
            'Precisão': [self.results[m]['precision'] for m in self.results.keys()],
            'Revocação': [self.results[m]['recall'] for m in self.results.keys()],
            'F1-Score': [self.results[m]['f1'] for m in self.results.keys()],
            'AUC-ROC': [self.results[m]['auc'] for m in self.results.keys()]
        })
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Comparação de Desempenho dos Modelos (Conjunto de Teste)',
                     fontsize=16, fontweight='bold')
        metrics = ['Acurácia', 'Precisão', 'Revocação', 'F1-Score', 'AUC-ROC']
        colors = ['steelblue', 'coral', 'lightgreen']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            bars = ax.bar(df['Modelo'], df[metric], color=colors)
            ax.set_ylabel(metric, fontweight='bold')
            ax.set_title(metric, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(alpha=0.3, axis='y')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        fig.delaxes(axes[1, 2])
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/10_comparacao_modelos.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_model_metrics_by_algorithm(self):
        print("\n[Gerando] Métricas por algoritmo")
        metric_names = {
            'accuracy': 'Acurácia',
            'precision': 'Precisão',
            'recall': 'Revocação',
            'f1': 'F1-Score',
            'auc': 'AUC-ROC',
        }
        for model_name, result in self.results.items():
            labels = list(metric_names.values())
            values = [result[key] for key in metric_names.keys()]
            fig, ax = plt.subplots(figsize=(9, 6))
            bars = ax.bar(labels, values, color='#4C78A8')
            ax.set_title(f'Métricas no Teste - {model_name}', fontweight='bold')
            ax.set_ylabel('Valor')
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3, axis='y')
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f'{value:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=10,
                )
            plt.tight_layout()
            algorithm_dir = self._algorithm_output_dir(model_name)
            plt.savefig(algorithm_dir / 'metricas.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_confusion_matrices(self):
        print("\n[Gerando] Matrizes de Confusão por algoritmo")
        for model_name in ['Random Forest', 'XGBoost', 'Rede Neural']:
            fig, ax = plt.subplots(figsize=(7, 6))
            cm = confusion_matrix(self.y_test, self.results[model_name]['y_pred'])
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=list(CLASS_LABELS.values()),
                yticklabels=list(CLASS_LABELS.values()),
                ax=ax,
                cbar_kws={'label': 'Freq'},
            )
            ax.set_title(f'Matriz de Confusão - {model_name}', fontweight='bold')
            ax.set_ylabel('Verdadeiro')
            ax.set_xlabel('Predito')
            plt.tight_layout()
            algorithm_dir = self._algorithm_output_dir(model_name)
            plt.savefig(algorithm_dir / 'matriz_confusao.png', dpi=300, bbox_inches='tight')
            plt.close()

    def plot_feature_importance(self):
        print("\n[Gerando] Importância das Features por algoritmo")
        feature_importances = {
            'Random Forest': self.rf_best.feature_importances_,
            'XGBoost': self.xgb_best.feature_importances_,
        }

        for model_name, importances in feature_importances.items():
            importance_df = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': importances,
            }).sort_values('importance', ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(9, 7))
            ax.barh(range(len(importance_df)), importance_df['importance'].values, color='#4C78A8')
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'].values, fontsize=9)
            ax.invert_yaxis()
            ax.set_title(f'Importância das Features - {model_name}', fontweight='bold')
            ax.set_xlabel('Importância')
            ax.grid(alpha=0.3, axis='x')
            plt.tight_layout()
            algorithm_dir = self._algorithm_output_dir(model_name)
            plt.savefig(algorithm_dir / 'importancia_features.png', dpi=300, bbox_inches='tight')
            plt.close()

    def save_models(self):
        print("\n[Salvando] Modelos treinados")
        joblib.dump(self.rf_best, f'{self.models_dir}/random_forest_best.pkl')
        joblib.dump(self.xgb_best, f'{self.models_dir}/xgboost_best.pkl')
        joblib.dump(self.mlp_best, f'{self.models_dir}/neural_network_best.pkl')
        stale_scaler = self.models_dir / 'scaler.pkl'
        if stale_scaler.exists():
            stale_scaler.unlink()

    def generate_summary_report(self):
        print("\n[Gerando] Relatório")
        best = max(self.results.keys(), key=lambda x: self.results[x]['f1'])
        report = f"""{'='*80}
RELATÓRIO DE TREINAMENTO DE MODELOS
Dataset: Diabetes 130-US Hospitals 1999-2008
{'='*80}

1. ALGORITMOS IMPLEMENTADOS

1.1 RANDOM FOREST
   - Melhor configuração: {format_params(self.best_params['Random Forest'])}
   - Justificativa: Robusto, bom com dados desbalanceados, interpretável

1.2 XGBOOST (Gradient Boosting)
   - Melhor configuração: {format_params(self.best_params['XGBoost'])}
   - Justificativa: Estado-da-arte, muito eficaz em dados tabulares

1.3 REDE NEURAL (MLP)
   - Melhor configuração: {format_params(self.best_params['Rede Neural'])}
   - Justificativa: Captura não-linearidades, complementa métodos baseados em árvores

2. DIVISÃO DOS DADOS
   - Treino/CV: 80% ({self.X_train.shape[0]} amostras)
   - Teste: 20% ({self.X_test.shape[0]} amostras)
   - Estratégia: divisão estratificada para manter proporção de classes

3. METODOLOGIA
   - Busca de hiperparâmetros: GridSearchCV
   - Validação cruzada: StratifiedKFold com {CV_FOLDS} folds
   - Métrica de seleção: F1-score ponderado (weighted)
   - Teste: usado apenas na avaliação final dos melhores modelos
   - Reprodutibilidade: seed fixo (random_state=42)
   - Resultado detalhado da busca: output/hyperparameter_search_results.csv
   - Melhores hiperparâmetros: output/best_hyperparameters.csv
   - Classification report por classe: output/classification_report_por_classe.csv

4. ARTEFATOS GRÁFICOS GERADOS
{'-'*80}
   - Distribuição das classes: output/09_distribuicao_classes.png
   - Comparação geral dos modelos: output/10_comparacao_modelos.png
   - Métricas por algoritmo: output/algoritmos/<algoritmo>/metricas.png
   - Matrizes de confusão por algoritmo: output/algoritmos/<algoritmo>/matriz_confusao.png
   - Importância das features por algoritmo: output/algoritmos/<algoritmo>/importancia_features.png
   - Resultados do GridSearchCV por algoritmo: output/algoritmos/<algoritmo>/gridsearchcv_resultados.png

5. RESULTADOS NO CONJUNTO DE TESTE
{'-'*80}
"""

        for model_name in sorted(self.results.keys()):
            m = self.results[model_name]
            report += f"\n{model_name}:\n"
            report += f"->   Acurácia:  {m['accuracy']:.4f}\n"
            report += f"->   Precisão:  {m['precision']:.4f}\n"
            report += f"->   Revocação: {m['recall']:.4f}\n"
            report += f"->   F1-Score:  {m['f1']:.4f}\n"
            report += f"->   AUC-ROC:   {m['auc']:.4f}\n"

        report += f"""
6. CONCLUSÃO
{'-'*80}
   Melhor Modelo: {best} (F1={self.results[best]['f1']:.4f})

   - Hiperparâmetros selecionados por GridSearchCV com validação cruzada estratificada
   - Todos os modelos foram avaliados sob a mesma divisão de teste
   - Ensemble methods tendem a ser competitivos para dados tabulares
   - Rede Neural oferece complementaridade, mas requer normalização e ajuste cuidadoso

{'='*80}"""

        with open(f'{self.output_dir}/RELATORIO_MODELOS.txt', 'w', encoding='utf-8') as f:
            f.write(report)

    def run_full_training(self):
        print("\n" + "="*80)
        print("TREINAMENTO DE MODELOS DE CLASSIFICAÇÃO")
        print("="*80)
        self.train_random_forest()
        self.train_xgboost()
        self.train_neural_network()
        self.evaluate_all_models_on_test()
        self.save_hyperparameter_search_results()
        self.save_best_hyperparameters_table()
        self.save_classification_reports()
        self.plot_class_distribution()
        self.plot_comparison_results()
        self.plot_model_metrics_by_algorithm()
        self.plot_grid_search_results()
        self.plot_confusion_matrices()
        self.plot_feature_importance()
        self.save_models()
        self.generate_summary_report()
        print("\n" + "="*80)
        print("TREINAMENTO CONCLUÍDO!")
        print("="*80)


def main():
    csv_path = PROJECT_ROOT / 'dataset' / 'diabetes_processed.csv'
    try:
        trainer = ModelTrainer(csv_path)
        trainer.run_full_training()
    except Exception as e:
        print(f"Erro: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()