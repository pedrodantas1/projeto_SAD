import pandas as pd
import numpy as np

# Carregar os dados
df = pd.read_csv("censo_2022_alfabetizacao.csv")

# Qtd de linhas com população nula
linhas_com_vulnerabilidade_nula = df[df['populacao'].isnull()].shape[0]
print(f"Quantidade de linhas com população nula: {linhas_com_vulnerabilidade_nula}")

# Remover registros com população nula
df = df.dropna(subset=["populacao"])

# Converter alfabetização para binário (1 = alfabetizado, 0 = não)
df["alfabetizados"] = df["alfabetizacao"].apply(lambda x: 1 if x == "Alfabetizadas" else 0)

# Agrupar população por grupo
grouped = df.groupby(["id_municipio", "cor_raca", "sexo", "grupo_idade", "alfabetizados"], as_index=False)["populacao"].sum()

# Pivotar para separar alfabetizados e não alfabetizados
pivot = grouped.pivot_table(index=["id_municipio", "cor_raca", "sexo", "grupo_idade"],
                            columns="alfabetizados",
                            values="populacao",
                            fill_value=0).reset_index()

# Renomear colunas
pivot.columns.name = None
pivot.columns = ["id_municipio", "cor_raca", "sexo", "grupo_idade", "pop_nao_alfabetizada", "pop_alfabetizada"]

# Total de população
pivot["populacao_total"] = pivot["pop_nao_alfabetizada"] + pivot["pop_alfabetizada"]



# pivot.to_csv("censo_2022_alfabetizacao_pivot.csv", index=False)