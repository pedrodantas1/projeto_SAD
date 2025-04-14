import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import *
from sklearn.model_selection import train_test_split



# Configurando visualizações
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.unicode_minus'] = False


# Carregando e preparando os dados
data = pd.read_csv("Alfabetizacao_Data_Mining.csv")

# Agregando dados por município primeiro
municipios_agregados = data.groupby(['nome_municipio', 'uf']).agg({
    'pop_alfabetizada': 'sum',
    'populacao_total': 'sum'
}).reset_index()

# Criando agregações demográficas corretas
for caracteristica in ['grupo_idade', 'cor_raca', 'sexo']:
    # Pivot table para somar a população por característica
    pivot = pd.pivot_table(
        data=data,
        values='populacao_total',
        index=['nome_municipio', 'uf'],
        columns=caracteristica,
        aggfunc='sum',
        fill_value=0
    )
    
    # Juntando com dados principais
    municipios_agregados = municipios_agregados.merge(
        pivot, on=['nome_municipio', 'uf']
    )
    
    # Criando proporções
    for coluna in pivot.columns:
        novo_nome = f'prop_{caracteristica}_{coluna}'.replace(' ', '_')
        municipios_agregados[novo_nome] = municipios_agregados[coluna] / municipios_agregados['populacao_total']

# Calculando taxa de alfabetização por município
municipios_agregados['taxa_alfabetizacao'] = (municipios_agregados['pop_alfabetizada'] / 
                                             municipios_agregados['populacao_total'] * 100)

# Calculando índice complexo com as novas proporções (ajustando os nomes das colunas)
municipios_agregados['indice_complexo'] = (
    municipios_agregados['taxa_alfabetizacao'] * 0.80 +
    municipios_agregados['prop_grupo_idade_65_anos_ou_mais'].fillna(0) * 5 +
    municipios_agregados['prop_grupo_idade_15_a_19_anos'].fillna(0) * 5 +
    (municipios_agregados['prop_cor_raca_Preta'].fillna(0) + 
     municipios_agregados['prop_cor_raca_Parda'].fillna(0)) * 4 +
     municipios_agregados['prop_cor_raca_Indígena'].fillna(0) * 6
)

# Definindo vulnerabilidade
municipios_agregados['vulnerabilidade_educacional'] = pd.cut(
    municipios_agregados['indice_complexo'],
    bins=[0, 60, 80, 100],
    labels=['Alta', 'Média', 'Baixa']
)

print("\nDistribuição das classes de vulnerabilidade:")
print(municipios_agregados['vulnerabilidade_educacional'].value_counts(normalize=True).round(3) * 100, '%')

# Salvar municípios agregados em um novo arquivo CSV
municipios_agregados.to_csv('municipios_agregados.csv', index=False, encoding='utf-8-sig')
print("\nDados salvos em 'municipios_agregados.csv'")


# Remover linhas com valores NaN em vulnerabilidade_educacional
municipios_agregados.dropna(subset=['vulnerabilidade_educacional'], inplace=True)

# Relatório Descritivo e Visualizações permanecem os mesmos...
print("=== RELATÓRIO DESCRITIVO DA BASE ===")
print("\nEstatísticas da Taxa de Alfabetização:")
print(municipios_agregados['taxa_alfabetizacao'].describe())

# # Análise detalhada de outliers
# outliers_df = municipios_agregados[['nome_municipio', 'uf', 'taxa_alfabetizacao']].copy()
# Q1 = outliers_df['taxa_alfabetizacao'].quantile(0.25)
# Q3 = outliers_df['taxa_alfabetizacao'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR

# # Identificando municípios outliers
# outliers_baixos = outliers_df[outliers_df['taxa_alfabetizacao'] < lower_bound].sort_values('taxa_alfabetizacao')
# outliers_altos = outliers_df[outliers_df['taxa_alfabetizacao'] > upper_bound].sort_values('taxa_alfabetizacao', ascending=False)

# print("\nExemplos de Outliers Baixos (5 municípios com menor taxa):")
# print(outliers_baixos[['nome_municipio', 'uf', 'taxa_alfabetizacao']].head())

# print("\nExemplos de Outliers Altos (5 municípios com maior taxa):")
# print(outliers_altos[['nome_municipio', 'uf', 'taxa_alfabetizacao']].head())

# Visualizações focadas na taxa de alfabetização
plt.figure(figsize=(10, 6))
sns.boxplot(data=municipios_agregados[['taxa_alfabetizacao']])
plt.title('Distribuição da Taxa de Alfabetização')
# plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=municipios_agregados, x='taxa_alfabetizacao', hue='vulnerabilidade_educacional')
plt.title('Distribuição da Taxa de Alfabetização por Nível de Vulnerabilidade')
# plt.show()

# Removendo as colunas não necessárias para o modelo
colunas_remover = [
    '15 a 19 anos', '20 a 24 anos', '25 a 34 anos',  # Contagens brutas de idade
    '35 a 44 anos', '45 a 54 anos', '55 a 64 anos',
    '65 anos ou mais',
    'Amarela', 'Branca', 'Indígena', 'Parda', 'Preta',  # Contagens brutas de raça
    'Homens', 'Mulheres',                      # Contagens brutas de sexo
    'indice_complexo'                          # Usado para criar o target
]

data_model = municipios_agregados.drop(colunas_remover, axis=1)

# Verificando as colunas mantidas
print("\nColunas mantidas para o modelo:", data_model.columns.tolist())

# Continuando com o split e setup do modelo...
train, test = train_test_split(data_model, test_size=0.1, random_state=42)

# Configurando o experimento no PyCaret
exp = setup(
    data=train,
    target='vulnerabilidade_educacional',
    numeric_features=['taxa_alfabetizacao'],
    categorical_features=[col for col in train.columns if col.startswith('prop_')],
    ignore_features=['nome_municipio', 'uf','pop_alfabetizada', 'populacao_total'],  # Mantém as colunas mas ignora no treinamento
    transformation=True,
    normalize=True,
    fix_imbalance=True,
    fold=10,
    session_id=42
)

# Comparando DT e RF (2 melhores)
print("\n=== COMPARAÇÃO DE MODELOS DE CLASSIFICAÇÃO ===")
models_comparison = compare_models(
    include=['dt', 'rf'],
    fold=10,
    sort='F1',
    n_select=1
)

# Avaliando cada modelo individualmente
# for model in ['dt', 'rf']:
#     print(f"\n=== AVALIAÇÃO DETALHADA - {model.upper()} ===")
#     clf = create_model(model)
    
#     # Métricas detalhadas
#     print("\nMétricas de Cross-Validation (10-fold):")
#     metrics = pull()  # Obtém as métricas do último modelo criado
#     print(f"Acurácia: {metrics.loc[model, 'Accuracy'].round(4)}")
#     print(f"Precisão: {metrics.loc[model, 'Precision'].round(4)}")
#     print(f"Recall: {metrics.loc[model, 'Recall'].round(4)}")
#     print(f"F1-Score: {metrics.loc[model, 'F1'].round(4)}")
    
#     # Matriz de confusão
#     print("\nMatriz de Confusão:")
#     plot_model(clf, plot='confusion_matrix')

# Criando e ajustando o melhor modelo
if isinstance(models_comparison, list):
    final_model = models_comparison[0]
else:
    final_model = models_comparison
tuned_model = tune_model(final_model, optimize='Accuracy')

# Prediçoes com dados de treino para avaliação inicial
print("\n=== PREVISÕES COM DADOS DE TREINO ===")
train_predictions = predict_model(tuned_model, data=train)
print(train_predictions.head())

# # Avaliando o modelo final
# print("\n=== AVALIAÇÃO DO MODELO FINAL ===")
# evaluate_model(tuned_model)

# # Plotando resultados importantes
# plot_model(tuned_model, plot='feature')
# plot_model(tuned_model, plot='confusion_matrix')
# plot_model(tuned_model, plot='auc')

# Finalizando o modelo
final_model_tuned = finalize_model(tuned_model)

# Fazendo previsões
print("\n=== FAZENDO PREVISÕES COM DADOS DE TESTE ===")
new_predictions = predict_model(tuned_model, data=test)
print(new_predictions.head()) 

# Salvando o modelo
save_model(final_model_tuned, 'modelo_vulnerabilidade_educacional')
print("\nModelo salvo como 'modelo_vulnerabilidade_educacional'")

# Colunas em test
print("\nColunas em test:", test.columns.tolist())

# Preparando relatórios dos dados de teste
test_results = pd.DataFrame({
    'Nome_Municipio': test['nome_municipio'],
    'UF': test['uf'],
    'Taxa_Alfabetizacao': test['taxa_alfabetizacao'].round(2),
    'Pop_Alfabetizada': test['pop_alfabetizada'],
    'Populacao_Total': test['populacao_total'],
    'Vulnerabilidade_Real': test['vulnerabilidade_educacional'],
    'Vulnerabilidade_Prevista': new_predictions['prediction_label'],
    'Score_Previsao': new_predictions['prediction_score'].round(4),
    'Previsao_Correta': test['vulnerabilidade_educacional'] == new_predictions['prediction_label'],
    
    # Proporções por idade
    'prop_15_19': test['prop_grupo_idade_15_a_19_anos'],
    'prop_20_24': test['prop_grupo_idade_20_a_24_anos'],
    'prop_25_34': test['prop_grupo_idade_25_a_34_anos'],
    'prop_35_44': test['prop_grupo_idade_35_a_44_anos'],
    'prop_45_54': test['prop_grupo_idade_45_a_54_anos'],
    'prop_55_64': test['prop_grupo_idade_55_a_64_anos'],
    'prop_65_mais': test['prop_grupo_idade_65_anos_ou_mais'],
    
    # Proporções por raça
    'prop_brancos': test['prop_cor_raca_Branca'],
    'prop_pretos': test['prop_cor_raca_Preta'],
    'prop_pardos': test['prop_cor_raca_Parda'],
    'prop_amarelos': test['prop_cor_raca_Amarela'],
    'prop_indigenas': test['prop_cor_raca_Indígena'],
    
    # Proporções por gênero
    'prop_homens': test['prop_sexo_Homens'],
    'prop_mulheres': test['prop_sexo_Mulheres']
})

# Salvando relatórios para Power BI
output_path = 'Dados_PowerBI'

# 1. Relatório Principal
test_results.to_csv(f'{output_path}/relatorio_teste_completo.csv', index=False, encoding='utf-8-sig')

# 2. Relatório de Acurácia por Região
acuracia_uf = test_results.groupby('UF').agg({
    'Previsao_Correta': 'mean',
    'Vulnerabilidade_Prevista': lambda x: x.value_counts().to_dict()
}).round(4)
acuracia_uf.to_csv(f'{output_path}/relatorio_acuracia_uf.csv', encoding='utf-8-sig')

# 3. Relatório Demográfico
# Relatórios demográficos por características
demographic_reports = {
    'idade': {
        'props': ['prop_15_19', 'prop_20_24', 'prop_25_34', 'prop_35_44',
                 'prop_45_54', 'prop_55_64', 'prop_65_mais'],
        'labels': ['15-19 anos', '20-24 anos', '25-34 anos', '35-44 anos',
                  '45-54 anos', '55-64 anos', '65+ anos']
    },
    'raca': {
        'props': ['prop_brancos', 'prop_pretos', 'prop_pardos',
                 'prop_amarelos', 'prop_indigenas'],
        'labels': ['Brancos', 'Pretos', 'Pardos', 'Amarelos', 'Indígenas']
    },
    'genero': {
        'props': ['prop_homens', 'prop_mulheres'],
        'labels': ['Homens', 'Mulheres']
    }
}

for categoria, info in demographic_reports.items():
    # Gerando relatório para cada característica demográfica
    for prop, label in zip(info['props'], info['labels']):
        # Criando bins para categorizar as proporções
        test_results[f'Faixa_{label}'] = pd.qcut(test_results[prop], q=3, 
                                                labels=['Baixa', 'Média', 'Alta'])
        
        demo_report = test_results.groupby(f'Faixa_{label}').agg({
            'Previsao_Correta': 'mean',
            'Score_Previsao': 'mean',
            'Vulnerabilidade_Prevista': lambda x: x.value_counts().to_dict(),
            prop: ['mean', 'min', 'max']  # Estatísticas da proporção
        }).round(4)
        
        demo_report.to_csv(f'{output_path}/relatorio_{categoria}_{label.lower().replace(" ", "_")}.csv', 
                          encoding='utf-8-sig')

print("\nRelatórios gerados para Power BI em:", output_path)

