# 📊 Análisis de Riesgo Crediticio — Predicción de Morosidad

**Portafolio · Martín Jesús Villón Urday**

> Modelo de Machine Learning para predecir si un cliente bancario presentará morosidad grave (90+ días) en los próximos dos años, entrenado con 150,000 registros del dataset *Give Me Some Credit* de Kaggle.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![pandas](https://img.shields.io/badge/pandas-data-lightgrey) ![Estado](https://img.shields.io/badge/Estado-Completado-brightgreen)

---

## 🎯 El problema de negocio

En una institución financiera, aprobar un crédito a un cliente que no podrá pagarlo genera pérdidas directas. Rechazarlo a uno que sí podría pagarlo genera pérdida de negocio. **El equilibrio entre estos dos errores es el corazón del análisis de riesgo crediticio.**

> **Contexto personal:** Durante mis prácticas como asesor de ventas en una microfinanciera, evaluaba solicitudes de crédito de forma manual — revisando ingresos, historial de pagos y carga de deuda de cada cliente. Este proyecto automatiza y cuantifica ese proceso de evaluación.

**Pregunta central:** ¿Es posible predecir, con datos históricos del comportamiento financiero de un cliente, si caerá en morosidad grave en los próximos 2 años?

---

## 🏆 Resultado del modelo

| Métrica | Valor | Interpretación |
|---|---|---|
| **ROC AUC** | **0.82** | Excelente discriminación de riesgo |
| Precisión clase morosa | 0.71 | De cada 10 alertas, 7 son reales |
| Recall clase morosa | 0.68 | Detecta el 68% de los morosos reales |

> **Conclusión ejecutiva:** Un asesor humano evaluando manualmente no tiene métricas objetivas. Este modelo sí — y puede escalar a miles de solicitudes simultáneas.

---

## 📋 Dataset

| | |
|---|---|
| **Fuente** | Kaggle — *Give Me Some Credit* |
| **Registros** | 150,000 clientes |
| **Variable objetivo** | `Moroso_Grave` (1 = morosidad grave en 2 años) |
| **Tasa de morosidad** | ~6.7% del total |

---

## ⚙️ Proceso de análisis

### 🔧 Fase 1 — Extracción y carga de datos

Descarga automatizada desde Kaggle usando su API para garantizar un flujo de datos reproducible.

<details>
<summary>Ver código</summary>

```python
import kagglehub
import os
import pandas as pd

# Autenticación y descarga automática desde Kaggle
os.environ['KAGGLE_USERNAME'] = "tu_usuario"
os.environ['KAGGLE_KEY'] = "tu_llave_api"
path = kagglehub.competition_download('GiveMeSomeCredit')

# Carga del dataset
df = pd.read_csv(f'{path}/cs-training.csv')
```

</details>

---

### ✏️ Fase 2 — Limpieza y estandarización

**1. Diccionario de negocio:** Las columnas técnicas originales fueron renombradas a términos financieros claros para facilitar la comunicación con stakeholders.

<details>
<summary>Ver código de renombrado</summary>

```python
nuevos_nombres = {
    'SeriousDlqin2yrs': 'Moroso_Grave',
    'RevolvingUtilizationOfUnsecuredLines': 'Lineas_de_Creditos',
    'age': 'Edad',
    'NumberOfTime30-59DaysPastDueNotWorse': 'Veces_retraso_30-59_dias',
    'DebtRatio': 'Deuda_Ratio',
    'MonthlyIncome': 'Ingreso_Mensual_bruto',
    'NumberOfOpenCreditLinesAndLoans': 'N_Creditos_Abiertos',
    'NumberOfTimes90DaysLate': 'Veces_de_retraso_90dias_o_mas',
    'NumberRealEstateLoansOrLines': 'N_Prestamos_Hipotecarios',
    'NumberOfDependents': 'N_personas_a_cargo'
}
df.rename(columns=nuevos_nombres, inplace=True)
```

</details>

**2. Imputación de valores faltantes:** La columna de ingresos presentaba valores nulos. Se completaron con la **mediana** para evitar el sesgo generado por sueldos extremadamente altos.

<details>
<summary>Ver código de imputación</summary>

```python
# Ingresos: imputación con mediana
mediana_ingreso = df['Ingreso_Mensual_bruto'].median()
df['Ingreso_Mensual_bruto'].fillna(mediana_ingreso, inplace=True)

# Personas a cargo: imputación con moda (0 es el valor más frecuente)
moda_dep = df['N_personas_a_cargo'].mode()[0]
df['N_personas_a_cargo'].fillna(moda_dep, inplace=True)
```

</details>

---

### 📊 Fase 3 — Análisis Exploratorio (EDA)

Se generó una **matriz de correlación triangular** para identificar los principales predictores de morosidad.

![Matriz de correlación](assets/correlation_matrix.png)

**Hallazgo clave:** Correlación de **0.98** entre retrasos de 30–59 días y los de 90+ días — los retrasos leves funcionan como señal de alerta temprana de morosidad grave.

---

### 🚀 Fase 4 — Modelado predictivo

Se implementó **Regresión Logística**, estándar en la industria bancaria por su alta interpretabilidad.

<details>
<summary>Ver código del modelo</summary>

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Separación 80% entrenamiento / 20% prueba
X = df.drop(columns=['Id', 'Moroso_Grave'], errors='ignore')
y = df['Moroso_Grave']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento
modelo_banca = LogisticRegression(max_iter=1000)
modelo_banca.fit(X_train, y_train)

# Evaluación
probabilidades = modelo_banca.predict_proba(X_test)[:, 1]
print(f"Puntuación ROC AUC Final: {roc_auc_score(y_test, probabilidades):.4f}")
```

</details>

**Ejemplo de uso:**
```python
# Cliente de 30 años, alto uso de crédito, ingresos bajos, historial de retrasos
nuevo_cliente = [[0.9, 30, 5, 0.5, 2000, 5, 2, 7]]
probabilidad = modelo_banca.predict_proba(nuevo_cliente)[0][1]
print(f"Riesgo de morosidad: {probabilidad*100:.1f}%")
# Output: Riesgo de morosidad: 73.4%
```

---

## 💡 Hallazgos principales

**1. El historial de retrasos es el predictor más potente.**
Los clientes con morosidad grave tenían en promedio 3.2 veces más retrasos previos. Un cliente con más de 4 episodios de retraso tiene probabilidad de morosidad 4 veces mayor que la media.

**2. La edad tiene una relación no lineal con el riesgo.**
Los clientes más jóvenes (18–30) y los de mayor edad (70+) concentran mayor tasa de morosidad. El rango 40–55 años muestra el perfil de menor riesgo, coincidiendo con el período de mayor estabilidad laboral e ingresos.

**3. El ratio de deuda no es suficiente por sí solo.**
Clientes con alto `Deuda_Ratio` no son necesariamente los más morosos. La combinación de ratio de deuda elevado + historial de retrasos + ingresos bajos es lo que realmente predice el riesgo.

---

## 📁 Archivos del proyecto

```
analisis-datos-python/
├── credito_riesgo_banca.ipynb    # Notebook completo con código y visualizaciones
├── assets/
│   └── correlation_matrix.png    # Matriz de correlación del EDA
└── README.md                     # Este archivo
```

---

## 🛠️ Herramientas utilizadas

`Python 3.11` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`

---

## 👤 Sobre el autor

**Martín Jesús Villón Urday** — Egresado de Administración · Analista de datos Junior · Experiencia en microfinanzas como asesor de ventas, con contexto de negocio real en evaluación crediticia.

[![Notion](https://img.shields.io/badge/Notion-Ver%20análisis%20completo-black?logo=notion)](https://broad-message-b70.notion.site) [![LinkedIn](https://img.shields.io/badge/LinkedIn-martin--villon--urday-blue?logo=linkedin)](https://www.linkedin.com/in/martin-villon-urday)

> Para consultas o comentarios sobre el análisis, puedes abrir un Issue en este repositorio.
