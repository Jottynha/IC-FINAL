<p align="center"> 
  <img src="imgs/logo_azul.png" alt="CEFET-MG" width="100px" height="100px">
</p>

<h1 align="center">
Inteligência Computacional
</h1>

<h3 align="center">
Predição de Readmissão Hospitalar de Pacientes Diabéticos Utilizando Técnicas de Inteligência Computacional
</h3>

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

</div>

---

<div align="justify">
<p><strong>Disciplina:</strong> Inteligência Computacional<br>
<strong>Instituição:</strong> Centro Federal de Educação Tecnológica de Minas Gerais (CEFET-MG) - Campus V Divinópolis<br>
<strong>Professor:</strong> Alisson Marques da Silva<br>
<strong>Projeto:</strong> Trabalho Final<br>
<strong>Alunos:</strong> João Pedro Rodrigues Silva e Pedro Augusto Gontijo Moura<br>
</div>

## Visão Geral

Este projeto investiga a capacidade preditiva de três algoritmos supervisionados na identificação de readmissões hospitalares de pacientes diabéticos em menos de 30 dias: Random Forest, XGBoost e Rede Neural Multilayer Perceptron (MLP). O estudo compara o desempenho dos modelos, analisa suas limitações e considera desafios típicos de bases médicas reais, como atributos categóricos, alta dimensionalidade, valores ausentes e desbalanceamento entre classes.

## Introdução

A readmissão hospitalar constitui um importante indicador da qualidade assistencial e da eficiência dos sistemas de saúde, estando frequentemente associada ao aumento dos custos operacionais, maior utilização de recursos hospitalares e pior prognóstico clínico para pacientes com doenças crônicas. Entre essas doenças, o diabetes mellitus destaca-se devido à elevada prevalência mundial e à recorrência de complicações clínicas que podem resultar em múltiplas hospitalizações.

Técnicas de Inteligência Computacional e aprendizado de máquina podem extrair padrões complexos de grandes volumes de dados clínicos e apoiar tarefas de classificação, predição de risco e tomada de decisão. Nesse contexto, a identificação precoce de pacientes com maior risco de readmissão pode apoiar ações preventivas e melhorar o planejamento assistencial.

## Base de Dados

O conjunto utilizado é o *Diabetes 130-US Hospitals for Years 1999-2008*, disponibilizado pelo UCI Machine Learning Repository. Ele reúne informações clínicas, demográficas e administrativas de atendimentos realizados em 130 hospitais dos Estados Unidos entre 1999 e 2008.

A tarefa foi formulada como classificação binária:

- Classe positiva (`1`): readmissão em menos de 30 dias.
- Classe negativa (`0`): readmissão após 30 dias ou ausência de readmissão.

O pré-processamento realiza o tratamento de valores ausentes, preenchimento por mediana ou moda, remoção de registros duplicados, One-Hot Encoding das variáveis categóricas, normalização dos nomes das colunas e remoção de atributos com variância inferior a `0,01`. O arquivo processado possui 101.766 registros e 140 colunas, sendo 139 atributos de entrada e uma variável alvo.

## Metodologia Computacional

Os dados são separados de forma estratificada em 80% para treinamento e 20% para teste. A seleção de hiperparâmetros é realizada pelo `GridSearchCV`, utilizando `StratifiedKFold` com três folds e F1-score da classe positiva como métrica de seleção.

Após a busca, o limiar de decisão de cada modelo é ajustado para maximizar o F1-score com previsões *out-of-fold* obtidas somente sobre o conjunto de treinamento. Os modelos são avaliados por acurácia, precisão, revocação, F1-score, ROC-AUC e PR-AUC.

Como os algoritmos possuem componentes aleatórios, a avaliação final é repetida com as sementes `7`, `21`, `42`, `84` e `126`. Os melhores hiperparâmetros são mantidos fixos, enquanto a divisão treino-teste e o ajuste do limiar são refeitos em cada execução. Os resultados finais são apresentados como média e desvio-padrão. O relatório por classe e as matrizes de confusão normalizadas também são agregados sobre as cinco execuções.

## Como Executar

```bash
# Clone o repositório
git clone https://github.com/Jottynha/IC-FINAL.git

# Acesse o diretório do projeto
cd IC-FINAL

# Instale as dependências
pip install -r requirements.txt

# Baixe e pré-processe os dados
python3 src/preprocess_diabetes.py

# Opcional: execute a análise exploratória
python3 src/exploratory_analysis.py

# Selecione os hiperparâmetros e treine os modelos
python3 src/model_training.py

# Execute a avaliação final com cinco sementes
python3 src/stability_experiment.py
```

O pré-processamento e parte da análise exploratória acessam o UCI Machine Learning Repository e, portanto, precisam de conexão com a internet. O `stability_experiment.py` reutiliza os hiperparâmetros salvos por `model_training.py`; por isso, os scripts devem ser executados nessa ordem.

A avaliação com cinco sementes também chama automaticamente o pós-processamento que gera as tabelas por classe e as matrizes de confusão médias. Caso `output/stability_results_by_seed.csv` já exista, esses artefatos podem ser regenerados em poucos segundos, sem treinar os modelos novamente:

```bash
python3 src/stability_postprocessing.py
```

## Artefatos Gerados

- `dataset/diabetes_processed.csv`: base pré-processada utilizada pelos experimentos.
- `models/`: melhores modelos e limiares de decisão salvos com Joblib.
- `output/resultados_binarios.csv`: métricas da execução base com a seed `42`.
- `output/best_hyperparameters.csv`: melhores configurações selecionadas pelo GridSearchCV.
- `output/gridsearchcv_resultados.csv`: resultados completos da busca de hiperparâmetros.
- `output/09_distribuicao_classes.png`: distribuição da variável alvo.
- `output/10_comparacao_modelos.png`: comparação dos modelos na execução base.
- `output/11_estabilidade_seeds.png`: médias e desvios das métricas nas cinco execuções.
- `output/analise_exploratoria/`: gráficos e relatório da análise exploratória.
- `output/algoritmos/<algoritmo>/`: métricas, curva Precision-Recall, GridSearchCV, importância de atributos quando aplicável e matrizes de confusão de cada modelo.
- `output/stability_results_by_seed.csv`: métricas detalhadas por modelo e seed.
- `output/stability_results_summary.csv`: médias e desvios das métricas finais.
- `output/stability_results_article_table.csv`: tabela geral formatada para o artigo.
- `output/stability_classification_report_by_seed.csv`: relatório por classe em cada execução.
- `output/stability_classification_report_summary.csv`: relatório por classe agregado.
- `output/stability_classification_report_article_table.csv`: relatório por classe formatado para o artigo.
- `output/stability_confusion_matrices_by_seed.csv`: células das matrizes reconstruídas por seed.
- `output/stability_confusion_matrices_summary.csv`: médias e desvios das matrizes.
- `output/algoritmos/<algoritmo>/matriz_confusao_media_normalizada.png`: matriz média normalizada em porcentagem.

Os arquivos `classification_report_por_classe.*` e `matriz_confusao.png` correspondem à execução base com a seed `42`. Para o artigo, devem ser usados os artefatos agregados cujo nome contém `stability` e as matrizes `matriz_confusao_media_normalizada.png`.

## Referências

[1] A. M. da Silva, *Trabalho Final*, CEFET-MG, Disciplina de Inteligência Computacional, 2026.

[2] B. Strack, J. P. DeShazo, C. Gennings, J. L. Olmo, S. Ventura, K. J. Cios e J. N. Clore, "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records", *BioMed Research International*, vol. 2014, pp. 1-11, 2014.

[3] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System", in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, San Francisco, USA, 2016, pp. 785-794.

[4] A. Esteva, A. Robicquet, B. Ramsundar et al., "A Guide to Deep Learning in Healthcare", *Nature Medicine*, vol. 25, pp. 24-29, 2019.

[5] L. Breiman, "Random Forests", *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.
