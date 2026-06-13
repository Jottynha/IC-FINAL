# Preprocessamento do Dataset Diabetes 130-US Hospitals for Years 1999-2008
# Dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_COL = "readmitted"


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

    df = df.replace("?", np.nan)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"  {col}: preenchido com mediana")

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            print(f"  {col}: preenchido com moda")

    return df


def remove_duplicates(df):
    print("\n--- Remocao de Duplicatas ---")
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    removed = initial_shape - df.shape[0]
    print(f"Registros removidos: {removed}")
    return df


def encode_categorical(df):
    print("\n--- Codificacao de Variaveis Categoricas (One-Hot Encoding) ---")
    categorical_cols = [
        col for col in df.select_dtypes(include=["object"]).columns
        if col != TARGET_COL
    ]

    if categorical_cols:
        for col in categorical_cols:
            print(f"  {col}: {df[col].nunique()} categorias")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
        print(f"Features apos One-Hot Encoding: {df.shape[1] - 1}")
    else:
        print("Nenhuma feature categorica encontrada")

    if TARGET_COL in df.columns and df[TARGET_COL].dtype == "object":
        target_mapping = {"<30": 0, ">30": 1, "NO": 2}
        df[TARGET_COL] = df[TARGET_COL].map(target_mapping)
        print(f"  {TARGET_COL}: codificado como {target_mapping}")

    if TARGET_COL in df.columns:
        feature_cols = [col for col in df.columns if col != TARGET_COL]
        df = df[feature_cols + [TARGET_COL]]

    df = sanitize_column_names(df)
    return df


def remove_low_variance_features(df, threshold=0.01):
    print(f"\n--- Remocao de Features com Baixa Variancia (threshold={threshold}) ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_drop = []

    for col in numeric_cols:
        if col == TARGET_COL:
            continue
        variance = df[col].var()
        if variance < threshold:
            cols_to_drop.append(col)

    if cols_to_drop:
        print(f"Features removidas: {cols_to_drop}")
        df = df.drop(cols_to_drop, axis=1)
    else:
        print("Nenhuma feature com baixa variancia encontrada")

    return df


def get_basic_statistics(df):
    print("\n--- Estatisticas Basicas ---")
    print(f"Shape final: {df.shape}")
    print(f"\nTipos de dados:\n{df.dtypes.value_counts()}")
    print(f"\nEstatisticas descritivas:\n{df.describe()}")


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
    print("PREPROCESSAMENTO CONCLUIDO COM SUCESSO")
    print("=" * 60)
    return df


def main():
    try:
        df = load_data()
        print("\nPrimeiras linhas do dataset:")
        print(df.head())

        df_processed = preprocess_pipeline(df)
        output_path = PROJECT_ROOT / "dataset" / "diabetes_processed.csv"
        output_path.parent.mkdir(exist_ok=True)
        df_processed.to_csv(output_path, index=False)
        print(f"\nDataset processado salvo em: {output_path}")
        return df_processed
    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")
        raise


if __name__ == "__main__":
    df_processed = main()
