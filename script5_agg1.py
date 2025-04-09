import pandas as pd
import os

# Carregar as tabelas do star schema
fato = pd.read_csv("star_schema/fato_alfabetizacao.csv")
dim_tempo = pd.read_csv("star_schema/dim_tempo.csv")
dim_municipio = pd.read_csv("star_schema/dim_municipio.csv")
dim_demografica = pd.read_csv("star_schema/dim_demografica.csv")

# Criar pasta para agregações se não existir
os.makedirs("star_schema/agregacoes", exist_ok=True)

# Juntar as dimensões com a tabela fato para facilitar a agregação
df_completo = pd.merge(fato, dim_tempo, on="id_tempo")
df_completo = pd.merge(df_completo, dim_municipio, on="id_municipio")
df_completo = pd.merge(df_completo, dim_demografica, on="id_demografica")

# Agregação por semestre e região
agregado_semestral = df_completo.groupby(['ano', 'semestre', 'regiao']).agg({
    'pop_alfabetizada': 'sum',
    'pop_nao_alfabetizada': 'sum',
    'populacao_total': 'sum'
}).reset_index()

# Calcular indicadores
agregado_semestral['taxa_alfabetizacao'] = (agregado_semestral['pop_alfabetizada'] / 
                                           agregado_semestral['populacao_total'] * 100).round(4)

# Calcular variação em relação ao semestre anterior
agregado_semestral = agregado_semestral.sort_values(['regiao', 'ano', 'semestre'])
agregado_semestral['variacao_taxa'] = agregado_semestral.groupby('regiao')['taxa_alfabetizacao'].diff().round(4)

# Salvar a tabela agregada
agregado_semestral.to_csv("star_schema/agregacoes/agregado_semestral_regiao.csv", index=False)

# Agregação por semestre, UF e sexo
agregado_uf_sexo = df_completo.groupby(['ano', 'semestre', 'uf', 'sexo']).agg({
    'pop_alfabetizada': 'sum',
    'pop_nao_alfabetizada': 'sum',
    'populacao_total': 'sum'
}).reset_index()

# Calcular taxa de alfabetização
agregado_uf_sexo['taxa_alfabetizacao'] = (agregado_uf_sexo['pop_alfabetizada'] / 
                                         agregado_uf_sexo['populacao_total'] * 100).round(4)

# Salvar a tabela agregada
agregado_uf_sexo.to_csv("star_schema/agregacoes/agregado_semestral_uf_sexo.csv", index=False)

print("Agregações semestrais criadas com sucesso:")
print("1. agregado_semestral_regiao.csv - Agregação por semestre e região")
print("2. agregado_semestral_uf_sexo.csv - Agregação por semestre, UF e sexo")