# Análisis de Riesgo Crediticio — Predicción de Morosidad
**Portafolio · Martín Jesús Villón Urday**

---

## El problema de negocio

En una institución financiera, aprobar un crédito a un cliente que no podrá pagarlo genera pérdidas directas. Rechazarlo a uno que sí podría pagarlo genera pérdida de negocio. El equilibrio entre estos dos errores es el corazón del análisis de riesgo crediticio.

**Pregunta central:** ¿Es posible predecir, con datos históricos del comportamiento financiero de un cliente, si caerá en morosidad grave en los próximos 2 años?

> **Contexto personal:** Durante mis prácticas como asesor de ventas en una microfinanciera, evaluaba solicitudes de crédito de forma manual — revisando ingresos, historial de pagos y carga de deuda de cada cliente. Este proyecto automatiza y cuantifica ese proceso de evaluación.

---

## Dataset

| Característica | Detalle |
|---|---|
| Fuente | Kaggle — Give Me Some Credit |
| Registros | 150,000 clientes |
| Variable objetivo | `Moroso_Grave` (1 = morosidad grave en 2 años) |
| Tasa de morosidad | ~6.7% del total |

---

## Variables analizadas

| Variable original | Nombre en español | Descripción |
|---|---|---|
| `SeriousDlqin2yrs` | `Moroso_Grave` | Objetivo: 1 si cayó en morosidad grave |
| `RevolvingUtilizationOfUnsecuredLines` | `Lineas_de_Credito` | % de crédito rotativo utilizado |
| `age` | `Edad` | Edad del cliente |
| `NumberOfTime30-59DaysPastDueNotWorse` | `Veces_retraso_30-59_dias` | Veces con retraso leve |
| `DebtRatio` | `Deuda_Ratio` | Deuda mensual / ingresos |
| `MonthlyIncome` | `Ingreso_Mensual_bruto` | Ingreso mensual declarado |
| `NumberOfOpenCreditLinesAndLoans` | `N_Creditos_Abiertos` | Número de créditos activos |
| `NumberRealEstateLoansOrLines` | `N_Prestamos_hipotecarios` | Préstamos con garantía hipotecaria |

---

## Proceso de análisis

### 1. Exploración inicial
- Revisión de distribuciones y valores nulos
- Agrupación de clientes por estado de morosidad para comparar perfiles medios
- Mapa de calor de correlaciones entre variables

### 2. Limpieza y filtros de negocio
- Eliminación de clientes con edad fuera del rango 18–100 años
- Remoción del 1% superior de `Deuda_Ratio` (valores físicamente imposibles)
- Creación de variable compuesta `Total_Veces_Retraso` (suma de retrasos 30–59 días y ≥90 días)

### 3. Modelado — Regresión Logística
- División 80% entrenamiento / 20% evaluación
- Modelo base: Regresión Logística (interpretable, estándar en banca)
- Evaluación con métricas apropiadas para datos desbalanceados

---

## Resultados del modelo

| Métrica | Valor |
|---|---|
| ROC AUC | **0.82** |
| Precisión clase morosa | 0.71 |
| Recall clase morosa | 0.68 |

El ROC AUC de 0.82 indica que el modelo tiene buena capacidad discriminatoria: en 8 de cada 10 comparaciones entre un cliente moroso y uno no moroso, el modelo asigna mayor probabilidad de riesgo al correcto.

### Ejemplo de uso del modelo
```python
# Cliente de 30 años, alto uso de crédito, ingresos bajos, historial de retrasos
nuevo_cliente = [[0.9, 30, 5, 0.5, 2000, 5, 2, 7]]
probabilidad = modelo_banca.predict_proba(nuevo_cliente)[0][1]

print(f"Riesgo de morosidad: {probabilidad*100:.1f}%")
# Output: Riesgo de morosidad: 73.4%
```

---

## Hallazgos principales

**1. El historial de retrasos es el predictor más potente**
Los clientes con morosidad grave tenían en promedio 3.2 veces más retrasos previos que los clientes al día. Un cliente con más de 4 episodios de retraso tiene probabilidad de morosidad 4 veces mayor que la media.

**2. La edad tiene una relación no lineal con el riesgo**
Los clientes más jóvenes (18–30 años) y los de mayor edad (70+) concentran mayor tasa de morosidad. El rango 40–55 años muestra el perfil de menor riesgo — coincide con el período de mayor estabilidad laboral e ingresos.

**3. El ratio de deuda no es suficiente por sí solo**
Clientes con alto `Deuda_Ratio` no necesariamente son los más morosos. La combinación de ratio de deuda elevado + historial de retrasos + ingresos bajos es lo que realmente predice el riesgo.

---

## Herramientas utilizadas

`Python 3.11` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`

---

## Archivos del proyecto

```
📁 analisis-datos-python/
├── credito_riesgo_banca.ipynb    # Notebook completo con código y visualizaciones
└── README.md                     # Este archivo
```

## Sobre el autor

**Martín Jesús Villón Urday**
Egresado de Administración y Sistemas · Analista de datos Junior
Experiencia en microfinanzas como asesor de ventas — contexto de negocio real en evaluación crediticia.

📂  https://www.notion.so/An-lisis-de-Riesgo-Crediticio-Predicci-n-de-Morosidad-Bancaria-35d987249e52807fa4f4cef5ba9932f8)) · 
💼  www.linkedin.com/in/martin-villon-urday

---
*Para consultas o feedback sobre el análisis, puedes abrir un Issue en este repositorio.*
