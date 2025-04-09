import pandas as pd
import numpy as np

df = pd.read_csv("fato_alfabetizacao_pre_pronto.csv")

# Contar quantas linhas tem ano 2023 e ano 2024 e printar com nome do ano antes
print(f"Ano 2023: {len(df[df['ano'] == 2023])}")
print(f"Ano 2024: {len(df[df['ano'] == 2024])}")

# Contar quantas linhas tem semestre 1 e semestre 2 e printar com nome do semestre antes
print(f"Semestre 1: {len(df[df['semestre'] == 1])}")
print(f"Semestre 2: {len(df[df['semestre'] == 2])}")

# Verificar colunas que sao float e converter para int
for col in df.columns:
    if df[col].dtype == "float64":
        df[col] = df[col].astype("int64")

# Colocar as colunas ano e semestre na treceira e quarta posicao sem saber as outras colunas
df = df[["id_municipio","ano", "semestre"] + [col for col in df.columns if col not in ["id_municipio","ano", "semestre"]]]

print(df.head())


# Salvar tabela
df.to_csv("fato_alfabetizacao_pre_pronto_v2.csv", index=False)