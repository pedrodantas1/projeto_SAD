import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from pycaret.classification import* 

 
data = pd.read_csv() 

print(data.head()) 
print(data.isnull().sum()) 


cols_to_int = ['accelerations', 'fetal_movement', 'uterine_contractions', 'light_decelerations', 'severe_decelerations', 'prolongued_decelerations']
data[cols_to_int] = data[cols_to_int].astype(int) 

data['accelerations'] = data['accelerations'] * 3600
data['fetal_movement'] = data['fetal_movement'] * 3600
data['uterine_contractions'] = data['uterine_contractions'] * 3600 

Total_por_classe = data['fetal_health'].value_counts()
print(Total_por_classe) 

plt.figure(figsize=(8, 6))
sns.countplot(x='fetal_health', data=data)
plt.title('Distribuição da Variável Alvo (Fetal Health)')
plt.xlabel('Estado de Saúde Fetal')
plt.ylabel('Número de Registros')
plt.xticks([0, 1, 2], ['Normal', 'Suspeito', 'Patológico'])
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='histogram_tendency', data=data)
plt.title('Distribuição da Coluna Histogram Tendency')
plt.xlabel('Tendência do Histograma')
plt.ylabel('Número de Registros')
plt.xticks([-1, 0, 1], ['Negativa', 'Simétrica', 'Positiva'])
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='severe_decelerations', data=data)
plt.title('Distribuição da Coluna Severe Decelerations')
plt.xlabel('Número de Desacelerações Severas por Hora')
plt.ylabel('Número de Registros')
plt.show()

print(data.describe()) 

data.hist(figsize=(20, 15))
plt.suptitle('Histogramas das Variáveis Numéricas', fontsize=20)
plt.show() 

corr_matrix = data.corr() 

plt.figure(figsize=(18, 15))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação entre as Variáveis')
plt.show() 

correlation_with_target = corr_matrix['fetal_health'].sort_values(ascending=False)
print(correlation_with_target) 

plt.figure(figsize=(10, 8))
correlation_with_target.drop('fetal_health').plot(kind='bar')
plt.title('Correlação com a Variável Alvo (Fetal Health)')
plt.xlabel('Variáveis')
plt.ylabel('Correlação')
plt.show() 

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15))
axes = axes.flatten()

for i, col in enumerate(['accelerations', 'light_decelerations', 'prolongued_decelerations', 'abnormal_short_term_variability', 'uterine_contractions', 'histogram_variance', 'histogram_mean', 'histogram_median']):
 sns.boxplot(x='fetal_health', y=col, data=data, ax=axes[i])
 axes[i].set_title(f'{col} por Estado de Saúde Fetal')
 axes[i].set_xticks([0, 1, 2])
 axes[i].set_xticklabels(['Normal', 'Suspeito', 'Patológico'])
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle('Comportamento das Variáveis por Estado de Saúde Fetal', fontsize=20)
plt.show()

train, test = train_test_split(data, test_size=0.1, random_state=42) 

reg = setup(data=train, target='fetal_health', transformation=True, fix_imbalance=True, session_id=1)


best_model = compare_models(sort="F1") 

lightgbm = create_model('lightgbm') 

tuned_lightgbm = tune_model(lightgbm, optimize="Accuracy") 

plot_model(lightgbm, plot='parameter') 
plot_model(lightgbm, plot='pipeline') 
plot_model(lightgbm, plot='feature') 
plot_model(lightgbm, plot='auc') 
plot_model(lightgbm, plot='confusion_matrix') 

predictions = predict_model(lightgbm)
print(predictions.head()) 

Modelo_final_fetos = finalize_model(lightgbm)
print(Modelo_final_fetos)

new_predictions = predict_model(Modelo_final_fetos, data=test)
print(new_predictions.head()) 

save_model(Modelo_final_fetos, 'Modelo_Saude_Fetal') 



