
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class ExploratoryAnalysis:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.output_dir = PROJECT_ROOT / 'output' / 'analise_exploratoria'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Dataset carregado: {self.df.shape[0]} linhas × {self.df.shape[1]} colunas")
    def analyze_basic_statistics(self):
        print("\n" + "="*80)
        print("ESTATÍSTICAS BÁSICAS DO DATASET")
        print("="*80)
        print(f"\nForma do dataset: {self.df.shape}")
        print(f"\nTipos de dados:\n{self.df.dtypes.value_counts()}")
        print(f"\nValores faltantes:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("-> Nenhum valor faltante encontrado")
        else:
            print(missing[missing > 0])
        print(f"\nEstatísticas descritivas (principais):")
        print(self.df.describe())
        print(f"\nRegistros duplicados: {self.df.duplicated().sum()}")
    def plot_distribution_overview(self):
        print("\n[Gerando] Plot 1: Distribuição Geral das Features")
        cols = self.df.columns[:12]
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        fig.suptitle('Distribuição das Features Numéricas (Principais)', fontsize=18, fontweight='bold', y=0.995)
        for idx, col in enumerate(cols):
            ax = axes[idx // 4, idx % 4]
            ax.hist(self.df[col], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_title(col, fontweight='bold', fontsize=11)
            ax.set_ylabel('Frequência', fontsize=9)
            ax.set_xlabel('Valor', fontsize=9)
            ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_distribuicao_features.png', dpi=300, bbox_inches='tight')
        plt.close()
    def plot_missing_values_pattern(self):
        print("\n[Gerando] Plot 2: Padrão de Valores Faltantes")
        from ucimlrepo import fetch_ucirepo
        diabetes = fetch_ucirepo(id=296)
        X = diabetes.data.features
        y = diabetes.data.targets
        df_original = pd.concat([X, y], axis=1)
        missing_data = df_original.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            missing_data.plot(kind='barh', ax=ax1, color='coral')
            ax1.set_xlabel('Quantidade de Valores Faltantes', fontweight='bold')
            ax1.set_title('Valores Faltantes por Feature (Dados Originais)', fontweight='bold', fontsize=12)
            ax1.grid(alpha=0.3)
            missing_percent = (missing_data / len(df_original) * 100).sort_values(ascending=False)
            missing_percent.plot(kind='barh', ax=ax2, color='lightcoral')
            ax2.set_xlabel('Percentual de Valores Faltantes (%)', fontweight='bold')
            ax2.set_title('Percentual de Valores Faltantes', fontweight='bold', fontsize=12)
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/02_valores_faltantes.png', dpi=300, bbox_inches='tight')
            plt.close()
    def plot_correlation_matrix(self):
        print("\n[Gerando] Plot 3: Matriz de Correlação")
        fig, ax = plt.subplots(figsize=(18, 14))
        # Selecionar apenas as primeiras 25 features para clareza
        cols_subset = self.df.columns[:25]
        corr_matrix = self.df[cols_subset].corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Matriz de Correlação (Primeiras 25 Features)', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_matriz_correlacao.png', dpi=300, bbox_inches='tight')
        plt.close()
    def plot_boxplot_quartiles(self):
        print("\n[Gerando] Plot 4: Boxplots (Quartis e Outliers)")
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        fig.suptitle('Boxplots de Features Numéricas - Detecção de Outliers', 
                     fontsize=16, fontweight='bold')
        cols = self.df.columns[:8]
        for idx, col in enumerate(cols):
            ax = axes[idx // 4, idx % 4]
            bp = ax.boxplot(self.df[col], vert=True, patch_artist=True, widths=0.5)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_linewidth(1.5)
            for whisker in bp['whiskers']:
                whisker.set_linewidth(1.5)
            for cap in bp['caps']:
                cap.set_linewidth(1.5)
            ax.set_title(col, fontweight='bold', fontsize=11)
            ax.set_ylabel('Valor', fontsize=9)
            ax.grid(alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_boxplots_outliers.png', dpi=300, bbox_inches='tight')
        plt.close()
    def plot_statistical_summary(self):
        print("\n[Gerando] Plot 5: Resumo Estatístico")
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Resumo Estatístico das Features', fontweight='bold', fontsize=16)
        means = self.df.mean().head(15)
        axes[0, 0].barh(range(len(means)), means.values, color='skyblue')
        axes[0, 0].set_yticks(range(len(means)))
        axes[0, 0].set_yticklabels(means.index, fontsize=9)
        axes[0, 0].set_xlabel('Média', fontweight='bold')
        axes[0, 0].set_title('Média das 15 Primeiras Features', fontweight='bold')
        axes[0, 0].grid(alpha=0.3, axis='x')
        stds = self.df.std().head(15)
        axes[0, 1].barh(range(len(stds)), stds.values, color='lightcoral')
        axes[0, 1].set_yticks(range(len(stds)))
        axes[0, 1].set_yticklabels(stds.index, fontsize=9)
        axes[0, 1].set_xlabel('Desvio Padrão', fontweight='bold')
        axes[0, 1].set_title('Desvio Padrão das 15 Primeiras Features', fontweight='bold')
        axes[0, 1].grid(alpha=0.3, axis='x')
        mins = self.df.min().head(15)
        axes[1, 0].barh(range(len(mins)), mins.values, color='lightgreen')
        axes[1, 0].set_yticks(range(len(mins)))
        axes[1, 0].set_yticklabels(mins.index, fontsize=9)
        axes[1, 0].set_xlabel('Valor Mínimo', fontweight='bold')
        axes[1, 0].set_title('Valor Mínimo das 15 Primeiras Features', fontweight='bold')
        axes[1, 0].grid(alpha=0.3, axis='x')
        maxs = self.df.max().head(15)
        axes[1, 1].barh(range(len(maxs)), maxs.values, color='lightyellow')
        axes[1, 1].set_yticks(range(len(maxs)))
        axes[1, 1].set_yticklabels(maxs.index, fontsize=9)
        axes[1, 1].set_xlabel('Valor Máximo', fontweight='bold')
        axes[1, 1].set_title('Valor Máximo das 15 Primeiras Features', fontweight='bold')
        axes[1, 1].grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_resumo_estatistico.png', dpi=300, bbox_inches='tight')
        plt.close()
    def plot_skewness_kurtosis(self):
        print("\n[Gerando] Plot 6: Skewness e Kurtosis")
        skewness = self.df.skew().head(16)
        kurtosis = self.df.kurtosis().head(16)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        colors_skew = ['red' if x < 0 else 'green' for x in skewness.values]
        ax1.barh(range(len(skewness)), skewness.values, color=colors_skew, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax1.set_yticks(range(len(skewness)))
        ax1.set_yticklabels(skewness.index, fontsize=10)
        ax1.set_xlabel('Skewness', fontweight='bold')
        ax1.set_title('Assimetria (Skewness) das Features\n(Vermelho: Esquerda | Verde: Direita)', 
                      fontweight='bold', fontsize=12)
        ax1.grid(alpha=0.3, axis='x')
        colors_kurt = ['purple' if x > 0 else 'orange' for x in kurtosis.values]
        ax2.barh(range(len(kurtosis)), kurtosis.values, color=colors_kurt, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax2.set_yticks(range(len(kurtosis)))
        ax2.set_yticklabels(kurtosis.index, fontsize=10)
        ax2.set_xlabel('Kurtosis', fontweight='bold')
        ax2.set_title('Curtose (Kurtosis) das Features\n(Roxo: Leptocúrtico | Laranja: Platicúrtico)', 
                      fontweight='bold', fontsize=12)
        ax2.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_skewness_kurtosis.png', dpi=300, bbox_inches='tight')
        plt.close()
    def plot_feature_variance(self):
        print("\n[Gerando] Plot 7: Variância Relativa das Features")
        variance = self.df.var().sort_values(ascending=False).head(20)
        variance_percent = (variance / variance.sum() * 100)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.bar(range(len(variance)), variance.values, color='steelblue', edgecolor='black')
        ax1.set_xticks(range(len(variance)))
        ax1.set_xticklabels(variance.index, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('Variância', fontweight='bold')
        ax1.set_title('Variância Absoluta - 20 Features com Maior Variação', fontweight='bold')
        ax1.grid(alpha=0.3, axis='y')
        ax2.bar(range(len(variance_percent)), variance_percent.values, color='coral', edgecolor='black')
        ax2.set_xticks(range(len(variance_percent)))
        ax2.set_xticklabels(variance_percent.index, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Percentual da Variância Total (%)', fontweight='bold')
        ax2.set_title('Variância Relativa (%) - 20 Features', fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_variancia_features.png', dpi=300, bbox_inches='tight')
        plt.close()
    def generate_summary_report(self):
        print("\n[Gerando] Relatório Resumido em Texto")
        
        report = f"""
{'='*80}
RELATÓRIO DE ANÁLISE EXPLORATÓRIA DE DADOS
Dataset: Diabetes 130-US Hospitals for Years 1999-2008
{'='*80}

1. INFORMAÇÕES GERAIS
{'-'*80}
   - Total de Registros: {self.df.shape[0]:,}
   - Total de Features: {self.df.shape[1]}
   - Tipos de Dados: {self.df.dtypes.value_counts().to_dict()}
   - Tamanho em Memória: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
   - Valores Faltantes: {self.df.isnull().sum().sum()}

2. CARACTERÍSTICAS ESTATÍSTICAS
{'-'*80}
   - Média Geral: {self.df.mean().mean():.4f}
   - Desvio Padrão Médio: {self.df.std().mean():.4f}
   - Mínimo Geral: {self.df.min().min():.4f}
   - Máximo Geral: {self.df.max().max():.4f}

3. DISTRIBUIÇÃO DE DADOS
{'-'*80}
   - Assimetria (Skewness) Média: {self.df.skew().mean():.4f}
   - Curtose (Kurtosis) Média: {self.df.kurtosis().mean():.4f}
   - Distribuição predominante: {'Simétrica' if abs(self.df.skew().mean()) < 1 else 'Assimétrica'}

4. VARIÂNCIA DOS DADOS
{'-'*80}
   - Variância Total: {self.df.var().sum():.4f}
   - Top 5 Features com Maior Variância:
"""
        
        top_var = self.df.var().sort_values(ascending=False).head(5)
        for idx, (col, var) in enumerate(top_var.items(), 1):
            report += f"     {idx}. {col}: {var:.4f}\n"
        
        report += f"""
5. CORRELAÇÃO
{'-'*80}
   - Correlação Máxima (excluindo diagonal): {self._get_max_correlation():.4f}
   - Correlação Mínima: {self._get_min_correlation():.4f}

6. NATUREZA DO DATASET
{'-'*80}
   Este dataset representa dados de 130 hospitais dos EUA cobrindo o período
   de 1999-2008. Cada registro representa um paciente hospitalizado com
   diagnóstico de diabetes. O dataset inclui:
   
   - Dados Clínicos: resultados de testes de laboratório, medicações
   - Dados Demográficos: idade, sexo, raça
   - Dados de Hospitalização: tempo de internação, tipo de admissão
   - Dados de Readmissão: indicador de readmissão em 30 dias
   
   Características Principais:
   - Dataset altamente desbalanceado em algumas variáveis
   - Múltiplas categorias com valores codificados numericamente
   - Ausência de valores faltantes após preprocessamento
   - Forte potencial para problemas de classificação

{'='*80}
FIM DO RELATÓRIO
{'='*80}
"""
        report_path = f'{self.output_dir}/RELATORIO_EDA.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(report)
    def _get_max_correlation(self):
        corr = self.df.corr().abs()
        corr_values = corr.values[np.triu_indices_from(corr.values, k=1)]
        return corr_values.max() if len(corr_values) > 0 else 0
    def _get_min_correlation(self):
        corr = self.df.corr().abs()
        corr_values = corr.values[np.triu_indices_from(corr.values, k=1)]
        return corr_values.min() if len(corr_values) > 0 else 0
    def run_full_analysis(self):
        print("\n" + "="*80)
        print("INICIANDO ANÁLISE EXPLORATÓRIA DE DADOS")
        print("="*80)
        self.analyze_basic_statistics()
        self.plot_distribution_overview()
        self.plot_missing_values_pattern()
        self.plot_correlation_matrix()
        self.plot_boxplot_quartiles()
        self.plot_statistical_summary()
        self.plot_skewness_kurtosis()
        self.plot_feature_variance()
        self.generate_summary_report()
def main():
    csv_path = PROJECT_ROOT / 'dataset' / 'diabetes_processed.csv'
    try:
        analysis = ExploratoryAnalysis(csv_path)
        analysis.run_full_analysis()
    except Exception as e:
        print(f"Erro durante a análise: {str(e)}")
        raise
if __name__ == "__main__":
    main()
