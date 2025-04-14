import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import *

# Configurando visualizações
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.unicode_minus'] = False


# Carregando e preparando os dados
data = pd.read_csv("censo_2022_alfabetizacao_pivot.csv")


# Garantindo que não temos valores negativos
data['pop_alfabetizada'] = data['pop_alfabetizada'].clip(lower=0)
data['populacao_total'] = data['populacao_total'].clip(lower=1)  # mínimo 1 para evitar divisão por zero

# Garantindo que pop_alfabetizada não é maior que populacao_total
data['pop_alfabetizada'] = data.apply(lambda x: min(x['pop_alfabetizada'], x['populacao_total']), axis=1)

data['taxa_alfabetizacao'] = (data['pop_alfabetizada'] / data['populacao_total']) * 100
data['vulnerabilidade_educacional'] = pd.cut(
    data['taxa_alfabetizacao'],
    bins=[0, 70, 85, 100],
    labels=['Alta', 'Média', 'Baixa']
)

# Remover linhas com valores NaN em vulnerabilidade_educacional
data.dropna(subset=['vulnerabilidade_educacional'], inplace=True)

# Relatório Descritivo e Visualizações permanecem os mesmos...
print("=== RELATÓRIO DESCRITIVO DA BASE ===")
print("\nEstatísticas da Taxa de Alfabetização:")
print(data['taxa_alfabetizacao'].describe())

# Identificação de Outliers apenas para taxa de alfabetização
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return len(outliers)

print(f"\nOutliers na Taxa de Alfabetização: {detect_outliers(data, 'taxa_alfabetizacao')}")

# Visualizações focadas na taxa de alfabetização
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['taxa_alfabetizacao']])
plt.title('Distribuição da Taxa de Alfabetização')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='taxa_alfabetizacao', hue='vulnerabilidade_educacional')
plt.title('Distribuição da Taxa de Alfabetização por Nível de Vulnerabilidade')
plt.show()

# Configurando o experimento no PyCaret
exp = setup(
    data=data,
    target='vulnerabilidade_educacional',
    numeric_features=['taxa_alfabetizacao'],
    categorical_features=['grupo_idade', 'cor_raca', 'sexo'],
    transformation=True,
    normalize=True,
    fix_imbalance=True,
    session_id=42
)

# Comparando todos os modelos
print("\n=== COMPARAÇÃO DE MODELOS ===")
best_model = compare_models(sort='F1', n_select=3)

# Criando e ajustando o melhor modelo
final_model = create_model(best_model)
tuned_model = tune_model(final_model, optimize='F1')

# Avaliando o modelo final
print("\n=== AVALIAÇÃO DO MODELO FINAL ===")
evaluate_model(tuned_model)

# Plotando resultados importantes
plot_model(tuned_model, plot='feature')
plot_model(tuned_model, plot='confusion_matrix')
plot_model(tuned_model, plot='auc')

# Finalizando o modelo
final_model_tuned = finalize_model(tuned_model)

# Salvando o modelo
save_model(final_model_tuned, 'modelo_vulnerabilidade_educacional_pycaret')
print("\nModelo salvo como 'modelo_vulnerabilidade_educacional_pycaret'")

# Exemplo de previsão
exemplo = data.iloc[0:1]
predicao = predict_model(final_model_tuned, data=exemplo)
print("\nExemplo de Previsão:")
print(predicao[['vulnerabilidade_educacional', 'prediction_label', 'prediction_score']])



