#Preprocessamento do Dataset Diabetes 130-US Hospitals for Years 1999-2008
#Dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')
def load_data():
    print("[Carregando dataset]")
    diabetes = fetch_ucirepo(id=296)
    X = diabetes.data.features
    y = diabetes.data.targets
    df = pd.concat([X, y], axis=1)
    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df
def handle_missing_values(df):
    print("\n--- Tratamento de Valores Faltantes ---")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Colunas com valores faltantes:\n{missing[missing > 0]}")
    else:
        print("Nenhum valor faltante encontrado (NaN)")
    df.replace('?', np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  {col}: preenchido com mediana")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  {col}: preenchido com moda")
    return df
def remove_duplicates(df):
    print("\n--- Remoção de Duplicatas ---")
    initial_shape = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed = initial_shape - df.shape[0]
    print(f"Registros removidos: {removed}")
    return df
def encode_categorical(df):
    print("\n--- Codificação de Variáveis Categóricas ---")
    categorical_cols = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} classes codificadas")
    return df
def remove_low_variance_features(df, threshold=0.01):
    print(f"\n--- Remoção de Features com Baixa Variância (threshold={threshold}) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_drop = []
    for col in numeric_cols:
        variance = df[col].var()
        if variance < threshold:
            cols_to_drop.append(col)
    if cols_to_drop:
        print(f"Features removidas: {cols_to_drop}")
        df.drop(cols_to_drop, axis=1, inplace=True)
    else:
        print("Nenhuma feature com baixa variância encontrada")
    return df
def get_basic_statistics(df):
    print("\n--- Estatísticas Básicas ---")
    print(f"Shape final: {df.shape}")
    print(f"\nTipos de dados:\n{df.dtypes.value_counts()}")
    print(f"\nEstatísticas descritivas:\n{df.describe()}")
def preprocess_pipeline(df):
    print("=" * 60)
    print("INICIANDO PREPROCESSAMENTO DO DATASET DIABETES")
    print("=" * 60)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = encode_categorical(df)
    df = remove_low_variance_features(df)
    get_basic_statistics(df)
    print("\n" + "=" * 60)
    print("PREPROCESSAMENTO CONCLUÍDO COM SUCESSO")
    print("=" * 60)
    return df
def main():
    try:
        df = load_data()
        print(f"\nPrimeiras linhas do dataset:")
        print(df.head())
        df_processed = preprocess_pipeline(df)
        output_path = '/home/joao/Projetos/IC-FINAL/dataset/diabetes_processed.csv'
        df_processed.to_csv(output_path, index=False)
        print(f"\nDataset processado salvo em: {output_path}")
        return df_processed
    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")
        raise
if __name__ == "__main__":
    df_processed = main()
