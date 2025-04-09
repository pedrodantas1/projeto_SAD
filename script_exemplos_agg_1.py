import pandas as pd
import matplotlib.pyplot as plt

# Carregar o arquivo
df = pd.read_csv("star_schema\\agregacoes\\disparidade_genero.csv")

# Filtrar apenas para o ano de 2024
df_2024 = df[df['ano'] == 2024]

# print("Dados de 2024:")
# print(df_2024)

# Filtrar apenas para a região Norte
df_norte = df[df['regiao'] == 'Norte']

# print("\nDados da região Norte:")
# print(df_norte)

# Selecionar uma região específica (exemplo: Nordeste)
regiao_especifica = 'Nordeste'
df_regiao = df[df['regiao'] == regiao_especifica]

# Ordenar por ano e semestre para visualizar a evolução
df_regiao = df_regiao.sort_values(['ano', 'semestre'])

print(f"\nEvolução da taxa de alfabetização feminina na região {regiao_especifica}:")
print(df_regiao[['ano', 'semestre', 'taxa_fem']])

# Criar um identificador único para cada período (ano+semestre)
df_regiao['periodo'] = df_regiao['ano'].astype(str) + '-' + df_regiao['semestre'].astype(str)

plt.figure(figsize=(10, 6))
plt.plot(df_regiao['periodo'], df_regiao['taxa_fem'], marker='o', linestyle='-')
plt.title(f'Evolução da Taxa de Alfabetização Feminina - Região {regiao_especifica}')
plt.xlabel('Período (Ano-Semestre)')
plt.ylabel('Taxa de Alfabetização (%)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"star_schema\\agregacoes\\evolucao_taxa_fem_{regiao_especifica}.png")
plt.close()

print(f"\nGráfico salvo como evolucao_taxa_fem_{regiao_especifica}.png")

# Filtrar para um período específico (exemplo: 2024, semestre 1)
ano_filtro = 2024
semestre_filtro = 1
df_periodo = df[(df['ano'] == ano_filtro) & (df['semestre'] == semestre_filtro)]

print(f"\nComparação entre regiões para {ano_filtro}/{semestre_filtro}:")
print(df_periodo[['regiao', 'taxa_masc', 'taxa_fem', 'diferenca_genero']])

# Para visualizar graficamente a diferença entre taxas masculinas e femininas
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = range(len(df_periodo))

plt.bar([i for i in index], df_periodo['taxa_masc'], bar_width, label='Masculino')
plt.bar([i + bar_width for i in index], df_periodo['taxa_fem'], bar_width, label='Feminino')

plt.xlabel('Região')
plt.ylabel('Taxa de Alfabetização (%)')
plt.title(f'Comparação de Taxas de Alfabetização por Gênero - {ano_filtro}/{semestre_filtro}')
plt.xticks([i + bar_width/2 for i in index], df_periodo['regiao'])
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(f"star_schema\\agregacoes\\comparacao_genero_{ano_filtro}_{semestre_filtro}.png")
plt.close()

print(f"\nGráfico de comparação salvo como comparacao_genero_{ano_filtro}_{semestre_filtro}.png")
