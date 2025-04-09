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

# 1. Desigualdade de alfabetização por cor/raça e região
agregado_raca_regiao = df_completo.groupby(['ano', 'semestre', 'regiao', 'cor_raca']).agg({
    'pop_alfabetizada': 'sum',
    'populacao_total': 'sum'
}).reset_index()

agregado_raca_regiao['taxa_alfabetizacao'] = (agregado_raca_regiao['pop_alfabetizada'] / 
                                             agregado_raca_regiao['populacao_total'] * 100).round(4)

# Calcular índice de desigualdade (desvio padrão das taxas por cor/raça em cada região)
desigualdade_raca = agregado_raca_regiao.groupby(['ano', 'semestre', 'regiao']).agg({
    'taxa_alfabetizacao': ['std', 'mean']
}).reset_index()

desigualdade_raca.columns = ['ano', 'semestre', 'regiao', 'indice_desigualdade_raca', 'taxa_media']
desigualdade_raca['indice_desigualdade_raca'] = desigualdade_raca['indice_desigualdade_raca'].round(4)
desigualdade_raca['taxa_media'] = desigualdade_raca['taxa_media'].round(4)

desigualdade_raca.to_csv("star_schema/agregacoes/desigualdade_raca_regiao.csv", index=False)

# 2. Concentração de analfabetismo por município (top 20 municípios com maior taxa)
concentracao_municipio = df_completo.groupby(['ano', 'semestre', 'uf', 'nome_municipio']).agg({
    'pop_nao_alfabetizada': 'sum',
    'populacao_total': 'sum'
}).reset_index()

concentracao_municipio['taxa_analfabetismo'] = (concentracao_municipio['pop_nao_alfabetizada'] / 
                                               concentracao_municipio['populacao_total'] * 100).round(4)

# Para cada período, encontrar os 20 municípios com maior taxa de analfabetismo
top_municipios = concentracao_municipio.sort_values(['ano', 'semestre', 'taxa_analfabetismo'], 
                                                   ascending=[True, True, False])
top_municipios['ranking'] = top_municipios.groupby(['ano', 'semestre']).cumcount() + 1
top_municipios = top_municipios[top_municipios['ranking'] <= 20]

top_municipios.to_csv("star_schema/agregacoes/concentracao_analfabetismo_municipios.csv", index=False)

# 3. Disparidade de gênero na alfabetização por região
disparidade_genero = df_completo.groupby(['ano', 'semestre', 'regiao', 'sexo']).agg({
    'pop_alfabetizada': 'sum',
    'populacao_total': 'sum'
}).reset_index()

disparidade_genero['taxa_alfabetizacao'] = (disparidade_genero['pop_alfabetizada'] / 
                                           disparidade_genero['populacao_total'] * 100).round(4)

# Pivotear para ter colunas por sexo
disparidade_pivot = disparidade_genero.pivot_table(
    index=['ano', 'semestre', 'regiao'],
    columns='sexo',
    values='taxa_alfabetizacao'
).reset_index()

# Renomear colunas
disparidade_pivot.columns.name = None
disparidade_pivot = disparidade_pivot.rename(columns={'Mulheres': 'taxa_fem', 'Homens': 'taxa_masc'})

# Calcular diferença entre taxas (M-F)
disparidade_pivot['diferenca_genero'] = (disparidade_pivot['taxa_masc'] - disparidade_pivot['taxa_fem']).round(4)

disparidade_pivot.to_csv("star_schema/agregacoes/disparidade_genero.csv", index=False)

# 4. Evolução temporal da alfabetização por faixa etária
evolucao_idade = df_completo.groupby(['ano', 'semestre', 'grupo_idade']).agg({
    'pop_alfabetizada': 'sum',
    'populacao_total': 'sum'
}).reset_index()

evolucao_idade['taxa_alfabetizacao'] = (evolucao_idade['pop_alfabetizada'] / 
                                       evolucao_idade['populacao_total'] * 100).round(4)

evolucao_idade = evolucao_idade.sort_values(['grupo_idade', 'ano', 'semestre'])
evolucao_idade['variacao'] = evolucao_idade.groupby('grupo_idade')['taxa_alfabetizacao'].diff().round(4)

evolucao_idade.to_csv("star_schema/agregacoes/evolucao_alfabetizacao_idade.csv", index=False)

# 5. Índice de vulnerabilidade educacional por UF
# (Combinação de alta taxa de analfabetismo e baixa variação positiva)
vulnerabilidade = df_completo.groupby(['ano', 'semestre', 'uf']).agg({
    'pop_alfabetizada': 'sum',
    'pop_nao_alfabetizada': 'sum',
    'populacao_total': 'sum'
}).reset_index()

vulnerabilidade['taxa_analfabetismo'] = (vulnerabilidade['pop_nao_alfabetizada'] / 
                                        vulnerabilidade['populacao_total'] * 100).round(4)

vulnerabilidade = vulnerabilidade.sort_values(['uf', 'ano', 'semestre'])
vulnerabilidade['variacao_taxa'] = vulnerabilidade.groupby('uf')['taxa_analfabetismo'].diff().round(4)

# Calcular índice de vulnerabilidade para todos os períodos (não apenas o último)
vulnerabilidade['indice_vulnerabilidade'] = (
    vulnerabilidade['taxa_analfabetismo'] - 
    5 * vulnerabilidade['variacao_taxa'].fillna(0)  # Peso maior para variação negativa (melhoria)
).round(4)

# Ordenar por período e índice de vulnerabilidade (do mais vulnerável para o menos)
vulnerabilidade_ordenada = vulnerabilidade.sort_values(['ano', 'semestre', 'indice_vulnerabilidade'], 
                                                      ascending=[True, True, False])

# Salvar todos os períodos
vulnerabilidade_ordenada.to_csv("star_schema/agregacoes/indice_vulnerabilidade_educacional_completo.csv", index=False)

# Calcular índice de vulnerabilidade (maior taxa + menor variação negativa = mais vulnerável)
# Criar uma coluna de período para ordenação correta (ano*10 + semestre)
vulnerabilidade['periodo_num'] = vulnerabilidade['ano']*10 + vulnerabilidade['semestre']

# Selecionar o período mais recente para cada UF
ultimo_periodo = vulnerabilidade.loc[vulnerabilidade.groupby('uf')['periodo_num'].idxmax()]

ultimo_periodo['indice_vulnerabilidade'] = (
    ultimo_periodo['taxa_analfabetismo'] - 
    5 * ultimo_periodo['variacao_taxa'].fillna(0)  # Peso maior para variação negativa (melhoria)
).round(4)

# Remover a coluna auxiliar antes de salvar
ultimo_periodo = ultimo_periodo.drop(columns=['periodo_num'])
ultimo_periodo = ultimo_periodo.sort_values('indice_vulnerabilidade', ascending=False)
ultimo_periodo.to_csv("star_schema/agregacoes/indice_vulnerabilidade_educacional.csv", index=False)

# 6. Correlação entre alfabetização e demografia
# Criar um agregado que mostra a correlação entre diferentes fatores demográficos
correlacao_demografica = pd.DataFrame()

# Para cada UF, calcular a correlação entre taxa de alfabetização e diferentes grupos demográficos
for uf in df_completo['uf'].unique():
    df_uf = df_completo[df_completo['uf'] == uf]
    
    # Agregação por características demográficas
    agg_demo = df_uf.groupby(['sexo', 'cor_raca', 'grupo_idade']).agg({
        'pop_alfabetizada': 'sum',
        'populacao_total': 'sum'
    }).reset_index()
    
    agg_demo['taxa_alfabetizacao'] = (agg_demo['pop_alfabetizada'] / 
                                     agg_demo['populacao_total'] * 100).round(4)
    
    # Criar variáveis dummy para características categóricas
    dummies_sexo = pd.get_dummies(agg_demo['sexo'], prefix='sexo')
    dummies_cor = pd.get_dummies(agg_demo['cor_raca'], prefix='cor')
    
    # Converter grupo_idade para numérico (extrair primeiro número)
    agg_demo['idade_num'] = agg_demo['grupo_idade'].str.extract('(\d+)').astype(float)
    
    # Combinar dados para correlação
    dados_corr = pd.concat([agg_demo[['taxa_alfabetizacao', 'idade_num']], 
                           dummies_sexo, dummies_cor], axis=1)
    
    # Calcular matriz de correlação
    corr_matrix = dados_corr.corr()['taxa_alfabetizacao'].drop('taxa_alfabetizacao')
    
    # Adicionar ao dataframe de resultados
    corr_row = pd.DataFrame([corr_matrix.values], 
                           columns=corr_matrix.index, 
                           index=[uf])
    
    correlacao_demografica = pd.concat([correlacao_demografica, corr_row])

# Resetar índice e adicionar como coluna
correlacao_demografica = correlacao_demografica.reset_index().rename(columns={'index': 'uf'})
correlacao_demografica = correlacao_demografica.round(4)

correlacao_demografica.to_csv("star_schema/agregacoes/correlacao_demografica_alfabetizacao.csv", index=False)

print("Agregações semestrais criadas com sucesso:")
print("1. agregado_semestral_regiao.csv - Agregação por semestre e região")
print("2. agregado_semestral_uf_sexo.csv - Agregação por semestre, UF e sexo")
print("3. desigualdade_raca_regiao.csv - Índice de desigualdade por cor/raça")
print("4. concentracao_analfabetismo_municipios.csv - Top 20 municípios com maior analfabetismo")
print("5. disparidade_genero.csv - Disparidade de alfabetização entre gêneros")
print("6. evolucao_alfabetizacao_idade.csv - Evolução da alfabetização por faixa etária")
print("7. indice_vulnerabilidade_educacional.csv - Índice de vulnerabilidade por UF")
print("8. correlacao_demografica_alfabetizacao.csv - Correlação entre fatores demográficos e alfabetização")