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
<strong>Projeto:</strong> "Trabalho Intermediário"<br>
<strong>Alunos:</strong> João Pedro Rodrigues Silva e Pedro Augusto Gontijo Moura<br>
</div>

## Visão Geral

O objetivo principal deste trabalho consiste em investigar a capacidade preditiva de diferentes algoritmos supervisionados aplicados à tarefa de classificação de readmissão hospitalar. Especificamente, busca-se comparar o desempenho de modelos baseados em árvores de decisão, métodos ensemble e arquiteturas neurais, analisando suas limitações, vantagens e adequação ao contexto clínico estudado. Adicionalmente, o projeto busca avaliar o impacto das etapas de preparação de dados, considerando desafios característicos de bases médicas reais, como atributos categóricos, elevada dimensionalidade, dados incompletos e possível desbalanceamento entre classes. 

## Introdução

A readmissão hospitalar constitui um importante indicador da qualidade assistencial e da eficiência dos sistemas de saúde, estando frequentemente associada ao aumento dos custos operacionais, maior utilização de recursos hospitalares e pior prognóstico clínico para pacientes com doenças crônicas. Entre essas doenças, o diabetes mellitus destaca-se devido à elevada prevalência mundial e à recorrência de complicações clínicas que podem resultar em múltiplas hospitalizações. Nesse contexto, a identificação precoce de pacientes com maior risco de readmissão representa uma estratégia relevante para apoiar ações preventivas e melhorar processos de tomada de decisão clínica.

Nos últimos anos, técnicas de Inteligência Computacional e aprendizado de máquina passaram a desempenhar papel importante em aplicações médicas devido à capacidade de extrair padrões complexos a partir de grandes volumes de dados clínicos. Essas abordagens têm sido empregadas em problemas relacionados à classificação, predição de risco, suporte à decisão clínica e modelagem prognóstica, contribuindo para sistemas mais eficientes e orientados por dados.

Este projeto propõe a aplicação e comparação de diferentes técnicas supervisionadas de aprendizado de máquina para a predição de readmissões hospitalares de pacientes diabéticos em até 30 dias após a alta médica. O estudo utiliza o conjunto de dados *Diabetes 130-US Hospitals for Years 1999–2008*, composto por informações clínicas, demográficas e administrativas coletadas em hospitais norte-americanos ao longo de aproximadamente uma década.

## Base de Dados

O conjunto de dados utilizado neste estudo corresponde ao dataset *Diabetes 130-US Hospitals for Years 1999–2008*, amplamente empregado em pesquisas relacionadas à mineração de dados clínicos e predição hospitalar. O dataset reúne registros provenientes de 130 hospitais dos Estados Unidos entre os anos de 1999 e 2008, contendo atributos relacionados ao perfil demográfico dos pacientes, histórico hospitalar, diagnósticos, exames laboratoriais e medicações administradas.

A variável alvo considerada corresponde à ocorrência de readmissão hospitalar em até 30 dias após a alta médica, caracterizando o problema como uma tarefa de classificação supervisionada.

Antes da etapa de modelagem, os dados passam por procedimentos de preparação envolvendo tratamento de valores ausentes, codificação de atributos categóricos, remoção de redundâncias e redução de atributos pouco informativos, visando melhorar a qualidade dos dados utilizados no treinamento dos modelos.

## Metodologia Computacional

Foram selecionadas três abordagens distintas de aprendizado supervisionado para compor os experimentos: Random Forest, XGBoost e Redes Neurais Multilayer Perceptron (MLP). A escolha dessas técnicas fundamenta-se em sua ampla utilização em problemas de classificação envolvendo dados tabulares e aplicações médicas.

O Random Forest foi selecionado devido à robustez frente a dados heterogêneos e à capacidade de lidar com grande quantidade de atributos. O XGBoost foi incluído devido ao elevado desempenho frequentemente reportado em tarefas de mineração de dados e classificação supervisionada. Por fim, Redes Neurais MLP foram utilizadas visando explorar a capacidade de modelagem de relações não lineares complexas presentes em dados clínicos. O pipeline experimental contempla etapas de coleta dos dados, preparação e pré-processamento, treinamento dos modelos, ajuste de hiperparâmetros e avaliação comparativa de desempenho.

## Como Executar o Código
Para executar o código deste projeto, siga as instruções abaixo:

```bash
# Clone o repositório
git clone Jottynha/IC-FINAL.git
# Acesse o diretório do projeto
cd IC-FINAL
# Instale as dependências necessárias
pip install -r requirements.txt
# Execute o preprocessamento dos dados
python3 src/preprocess_diabetes.py
# [Opcional] Execute a analise exploratória dos dados 
python3 src/exploratory_analysis.py
# Treine os modelos de aprendizado de máquina
python3 src/model_training.py
```
## Artefatos gerados
- `output/`: Diretório contendo os resultados dos experimentos, incluindo métricas de desempenho, gráficos e relatórios gerados durante a análise.
- `models/`: Diretório onde os modelos treinados são salvos para posterior avaliação e comparação.
- `output/analise_exploratoria/`: Subdiretório específico para os artefatos relacionados à análise exploratória dos dados, como gráficos de distribuição, correlações e relações iniciais.


## Referências

[1] A. M. da Silva, *Trabalho Final*, CEFET-MG, Disciplina de Inteligência Computacional, 2026.

[2] B. Strack, J. P. DeShazo, C. Gennings, J. L. Olmo, S. Ventura, K. J. Cios e J. N. Clore, “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records”, *BioMed Research International*, vol. 2014, pp. 1–11, 2014.

[3] T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System”, in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, San Francisco, USA, 2016, pp. 785–794.

[4] A. Esteva, A. Robicquet, B. Ramsundar et al., “A Guide to Deep Learning in Healthcare”, *Nature Medicine*, vol. 25, pp. 24–29, 2019.

[5] L. Breiman, “Random Forests”, *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

 
