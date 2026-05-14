!pip install kagglehub

!mkdir -p ~/.kaggle

!echo KGAT_e57ba34133bfa1268e51d8f06908b623 > ~/.kaggle/access_token

!chmod 600 ~/.kaggle/access_token


!kaggle competitions list

import kagglehub

# Download latest version
path = kagglehub.competition_download('GiveMeSomeCredit')

print("Path to competition files:", path)

import pandas as pd

# Definimos la ruta que te dio el sistema
ruta_datos = '/root/.cache/kagglehub/competitions/GiveMeSomeCredit'

# Cargamos el archivo de entrenamiento (train.csv)
df = pd.read_csv(f'{ruta_datos}/cs-training.csv')

df.head(10)

#Cambiar nombre

nuevos_nombres= {'Unnamed: 0':'Id',
                'SeriousDlqin2yrs':'Moroso_Grave',
                'RevolvingUtilizationOfUnsecuredLines':'Lineas_de_Creditos',
                'age':'edad',
                'NumberOfTime30-59DaysPastDueNotWorse':'Veces_de_retrazo_30-59_dias',
                'DebtRatio':'Deuda_Ratio',
                'MonthlyIncome':'Ingreso_Mensual_bruto',
                'NumberOfOpenCreditLinesAndLoans':'N.Creditos_Abiertos',
                'Veces_de_retrazo_>=90dias':'Veces_de_retrazo_>=90dias',
                'NumberRealEstateLoansOrLines':'N.Prestamos_hipotecarios',
                'NumberOfTime60-89DaysPastDueNotWorse':'N.veces_retrazo_60-89_dias',
                'NumberOfDependents':'N.personas_a_cargo'}

df.rename(columns=nuevos_nombres, inplace=True)


df.drop(columns=['Unnamed: 0', 'N.veces_retrazo_60-89_dias', 'N.personas_a_cargo'], inplace=True, errors='ignore')

# Esto te da el promedio y el conteo (N) por cada grupo
df.drop(columns='Id').groupby('Moroso_Grave').agg(['mean', 'count']).T

import seaborn as sns
import matplotlib.pyplot as plt

# 1. Creamos una copia del dataframe sin el Id para el gráfico
df_grafico = df.drop(columns='Id', errors='ignore')

# 2. Generamos el mapa de calor con los datos limpios
plt.figure(figsize=(10, 6))
sns.heatmap(df_grafico.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Calor de Relaciones Crediticias (Sin Id)')
plt.show()

import numpy as np

# Creamos una máscara para la mitad superior
mask = np.triu(np.ones_like(df_grafico.corr(), dtype=bool))

sns.heatmap(df_grafico.corr(), annot=True, mask=mask, cmap='coolwarm', fmt='.2f')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Creamos una cuadrícula para ver varias columnas a la vez
cols_interes = ['Edad', 'Deuda_Ratio', 'Ingreso_Mensual_bruto']
plt.figure(figsize=(15, 5))

for i, col in enumerate(cols_interes):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Distribución de {col}')

plt.tight_layout()
plt.show()

# Aplicando filtros lógicos de negocio
df = df[(df['Edad'] >= 18) & (df['Edad'] <= 100)]

# Limpiamos valores extremos en Deuda_Ratio (por ejemplo, quitando el 1% más alto)
percentil_99 = df['Deuda_Ratio'].quantile(0.99)
df = df[df['Deuda_Ratio'] < percentil_99]

# Creamos un indicador de "Retrasos Totales"
df['Total_Veces_Retraso'] = (df['Veces_de_retazo_30-59_dias'] + 
                             df['Veces_de_retrazo_>=90dias'])

# Verificamos cómo quedó nuestra base final
print(f"Registros finales para el modelo: {df.shape[0]}")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# 1. Definimos las características (X) y el objetivo (y)
X = df.drop(columns=['Id', 'Moroso_Grave'], errors='ignore')
y = df['Moroso_Grave']

# 2. Dividimos los datos: 80% para entrenar y 20% para evaluar el modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Entrenando con {len(X_train)} registros y probando con {len(X_test)}.")

# Creamos y entrenamos el modelo
modelo_banca = LogisticRegression(max_iter=1000)
modelo_banca.fit(X_train, y_train)

# Hacemos las predicciones con el 20% de datos que el modelo no conoce
predicciones = modelo_banca.predict(X_test)

# Evaluamos el desempeño
print("--- Informe de Clasificación ---")
print(classification_report(y_test, predicciones))

# Calculamos el área bajo la curva ROC (métrica estándar en Kaggle)
roc_score = roc_auc_score(y_test, modelo_banca.predict_proba(X_test)[:, 1])
print(f"Puntuación ROC AUC: {roc_score:.4f}")

# Ejemplo: Un cliente de 30 años, con pocos ingresos y muchos retrasos previos
nuevo_cliente = [[0.9, 30, 5, 0.5, 2000, 5, 2, 0, 7]] # Ajusta según tus columnas
probabilidad = modelo_banca.predict_proba(nuevo_cliente)[0][1]

print(f"El sistema indica un {probabilidad*100:.2f}% de riesgo de morosidad.")
