import random
import pandas as pd
import numpy as np

# Copiar base original
df_simulado = pd.read_csv("censo_2022_alfabetizacao_pivot.csv")

# Embaralhar os dados para distribuição equilibrada
df_embaralhado = df_simulado.sample(frac=1, random_state=42).reset_index(drop=True)

# Definir anos alternados balanceadamente
metade = len(df_embaralhado) // 2
anos = [2023] * metade + [2024] * (len(df_embaralhado) - metade)
random.shuffle(anos)

# Atribuir os anos e semestre aleatório
df_embaralhado["ano"] = anos
df_embaralhado["semestre"] = np.random.choice([1, 2], size=len(df_embaralhado))

# Renomear colunas para consistência com tabela fato
df_embaralhado.rename(columns={
    "populacao_total": "populacao_total",
    "pop_alfabetizada": "alfabetizados"
}, inplace=True)

print(df_embaralhado.head())

df_embaralhado.to_csv("fato_alfabetizacao_pre_pronto.csv", index=False)