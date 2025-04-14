import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import *

# Carregando o modelo e dados
modelo = load_model('modelo_vulnerabilidade_educacional')
dados_teste = pd.read_csv('Dados_PowerBI/relatorio_teste_completo.csv')

# Configurando estilo
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Matriz de Confusão
plt.figure()
confusion_matrix = pd.crosstab(dados_teste['Vulnerabilidade_Real'], 
                             dados_teste['Vulnerabilidade_Prevista'])
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão')
plt.savefig('Dados_PowerBI/matriz_confusao.png')

# 2. Distribuição de Vulnerabilidade por Estado
plt.figure()
dados_estado = dados_teste.groupby('UF')['Vulnerabilidade_Prevista'].value_counts().unstack()
dados_estado.plot(kind='bar', stacked=True)
plt.title('Distribuição de Vulnerabilidade por Estado')
plt.tight_layout()
plt.savefig('Dados_PowerBI/vulnerabilidade_por_estado.png')

# 3. Taxa de Alfabetização vs Vulnerabilidade
plt.figure()
sns.boxplot(data=dados_teste, x='Vulnerabilidade_Prevista', y='Taxa_Alfabetizacao')
plt.title('Taxa de Alfabetização por Nível de Vulnerabilidade')
plt.savefig('Dados_PowerBI/alfabetizacao_vs_vulnerabilidade.png')

# 4. Acurácia por Estado
plt.figure()
acuracia_estado = dados_teste.groupby('UF')['Previsao_Correta'].mean().sort_values()
acuracia_estado.plot(kind='bar')
plt.title('Acurácia do Modelo por Estado')
plt.tight_layout()
plt.savefig('Dados_PowerBI/acuracia_por_estado.png')

# 5. Métricas Gerais de Acurácia
metricas_gerais = {
    'Acurácia_Geral': dados_teste['Previsao_Correta'].mean(),
    'Total_Municipios': len(dados_teste),
    'Distribuicao_Classes': dados_teste['Vulnerabilidade_Prevista'].value_counts(normalize=True)
}

# 5. Distribuição Demográfica por Vulnerabilidade
plt.figure(figsize=(15, 6))
demographic_vars = ['prop_15_19', 'prop_65_mais', 'prop_pretos', 'prop_pardos', 'prop_indigenas']
demographic_labels = ['Jovens (15-19)', 'Idosos (65+)', 'Pretos', 'Pardos', 'Indígenas']

for i, (var, label) in enumerate(zip(demographic_vars, demographic_labels), 1):
    plt.subplot(1, 5, i)
    sns.boxplot(data=dados_teste, x='Vulnerabilidade_Prevista', y=var)
    plt.title(f'{label} por Vulnerabilidade')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Dados_PowerBI/demografia_por_vulnerabilidade.png')

# 6. Top 10 Municípios por Score de Previsão
plt.figure(figsize=(12, 6))
top_municipios = dados_teste.nlargest(10, 'Score_Previsao')
sns.barplot(data=top_municipios, x='Score_Previsao', y='Nome_Municipio')
plt.title('Top 10 Municípios - Maior Confiança na Previsão')
plt.tight_layout()
plt.savefig('Dados_PowerBI/top_municipios_score.png')

# 7. Distribuição dos Scores de Previsão
plt.figure()
sns.histplot(data=dados_teste, x='Score_Previsao', hue='Vulnerabilidade_Prevista', multiple="stack")
plt.title('Distribuição dos Scores de Previsão por Classe')
plt.savefig('Dados_PowerBI/distribuicao_scores.png')

# 8. Mapa de Calor das Correlações
plt.figure(figsize=(12, 10))
correlation_vars = ['Taxa_Alfabetizacao', 'prop_15_19', 'prop_65_mais', 
                   'prop_pretos', 'prop_pardos', 'prop_indigenas', 'Score_Previsao']
correlation_matrix = dados_teste[correlation_vars].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlações entre Variáveis Principais')
plt.tight_layout()
plt.savefig('Dados_PowerBI/correlacoes.png')

# 9. Comparação Real vs Previsto
plt.figure()
comparison_data = pd.crosstab(dados_teste['Vulnerabilidade_Real'], 
                            dados_teste['Vulnerabilidade_Prevista'], 
                            normalize='index') * 100
comparison_data.plot(kind='bar', stacked=True)
plt.title('Distribuição das Previsões por Classe Real (%)')
plt.ylabel('Porcentagem')
plt.legend(title='Previsto')
plt.tight_layout()
plt.savefig('Dados_PowerBI/comparacao_real_previsto.png')

print("Visualizações extras geradas com sucesso!")

# 10. Visualização alternativa para população indígena
plt.figure(figsize=(12, 6))

# Histograma com densidade
sns.histplot(data=dados_teste, x='prop_indigenas', hue='Vulnerabilidade_Prevista',
            multiple="stack", bins=50)
plt.title('Distribuição da Proporção de População Indígena')
plt.xlabel('Proporção de População Indígena')
plt.ylabel('Número de Municípios')

plt.tight_layout()
plt.savefig('Dados_PowerBI/distribuicao_indigenas.png')

# Top 10 municípios com maior população indígena
top_indigenas = dados_teste.nlargest(10, 'prop_indigenas')[['Nome_Municipio', 'UF', 'prop_indigenas', 'Vulnerabilidade_Prevista']]
top_indigenas.to_csv('Dados_PowerBI/top_municipios_indigenas.csv', index=False, encoding='utf-8-sig')

# Salvando métricas em CSV para Power BI
pd.DataFrame({
    'Metrica': ['Acurácia Geral', 'Total de Municípios'],
    'Valor': [f'{metricas_gerais["Acurácia_Geral"]:.2%}', metricas_gerais['Total_Municipios']]
}).to_csv('Dados_PowerBI/metricas_gerais.csv', index=False, encoding='utf-8-sig')

# Salvando distribuição das classes
pd.DataFrame(metricas_gerais['Distribuicao_Classes']).reset_index().rename(
    columns={'index': 'Classe', 'Vulnerabilidade_Prevista': 'Proporção'}
).to_csv('Dados_PowerBI/distribuicao_classes.csv', index=False, encoding='utf-8-sig')

print(f"\nAcurácia Geral do Modelo: {metricas_gerais['Acurácia_Geral']:.2%}")
print(f"Total de Municípios Analisados: {metricas_gerais['Total_Municipios']}")
print("\nDistribuição das Classes:")
print(metricas_gerais['Distribuicao_Classes'])