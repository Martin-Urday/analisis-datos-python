# analisis-datos-python
Análisis de Calidad en Proceso de Flotación Inversa de Hierro
Portafolio · Martín Jesús Villón Urday

El problema de negocio
Una planta minera de hierro genera toneladas de datos de sensores cada hora – flujos de aire, niveles de pulpa, dosis de reactivos – pero sin un análisis estructurado, ese volumen de información no mejora las decisiones operativas.

Pregunta central: ¿Qué variables del proceso de flotación tienen mayor influencia sobre la pureza del concentrado de hierro? ¿Y qué problemas de calidad tiene el conjunto de datos que impiden responder esa pregunta directamente?

Conjunto de datos
Importancia	Detalle
Fuente	Kaggle — Planta de flotación para procesos mineros
Período	Marzo – Mayo 2017 (64 días)
Tamaño original	220.000 filas × 23 columnas
Variable objetivo	% de concentrado de hierro
Problemas encontrados en los datos
Durante la auditoría del conjunto de datos identifiqué 5 problemas que harían fallar cualquier modelo entrenado directamente sobre el archivo original:

1. Separador decimal incorrecto
Todas las columnas numéricas usaban coma como separador decimal (formato europeo). Sin corrección, pandas las lee como texto y cualquier operación estadística falla silenciosamente.

2. Granularidad temporal mixta
El conjunto de datos tiene 220.000 filas pero solo 1.223 marcas de tiempo únicas. Existen dos frecuencias mezcladas sin documentarlo:

Variables de laboratorio (% Fe, % SiO₂, % Concentrado): se miden una vez por hora y se repiten en ~180 filas idénticas
Variables de proceso (flujos, niveles): sí fluctúan dentro de cada hora
Entrenar un modelo sin separar estas frecuencias genera fugas de datos y resultados engañosamente buenos.

3. Comportamiento bimodal en Air Flow (celdas 01–03)
Las columnas de flujo de aire de las celdas 1, 2 y 3 tienen entre el 34–35% de sus valores clasificados como valores atípicos por IQR. No son errores — el análisis con K-Means confirmó que cada celda opera en dos regímenes distintos: activa (20.3% del tiempo) y baja carga (79.7%). Eliminarlos hubiera destruido información clave operativa.

4. Picos en el flujo de almidón
El flujo de almidón presenta 9.269 picos (4,2% del conjunto de datos) con valores que caen hasta 367 g/t cuando la operación normal está entre 2.900–3.300 g/t. Se corrigieron mediante interpolación lineal por tiempo.

5. Gap temporal de 13 días
El proceso tiene una interrupción de 13 días y 7 horas sin registros. Crítico para no calcular características de retraso entre datos no contiguos.

Soluciones aplicadas
220,000 filas × 23 cols  →  limpieza  →  1,223 filas × 64 cols  →  feature engineering  →  1,221 filas × 82 cols
     (archivo original)                    (horario limpio)                                    (listo para modelo)
Problema	Solución aplicada
Decimal incorrecto	pd.read_csv(..., decimal=",")
Granularidad mixta	Separación + agregación horaria (media, std, p95)
Flujo de aire bimodal	K-Means k=2 → característica categórica por celda
Flujo de almidón en picos	Interpolación lineal temporal
Retraso temporal	Características t-1h y t-2h para 8 variables clave
Hallazgos principales
1. Las variables de aire mandan, no las de alimentación. La estimación lineal más fuerte con el concentrado final es el Air Flow de la columna 04 (r = −0.24) y columna 05 (r = +0.21). El % de hierro en la alimentación tiene una compensación de apenas −0.005. Esto sugiere que la operación de las celdas es más determinante que la calidad del mineral de entrada.

2. Las relaciones son no lineales Las bajas correlaciones lineales (ninguna supera 0.25) no significan que las variables sean irrelevantes — significan que un modelo de regresión lineal simple no capturará la dinámica real. Se recomienda XGBoost o Random Forest que detectan interacciones entre columnas.

3. Las celdas 01–03 tienen un patrón operativo diferente al resto. El 20% del tiempo estas celdas operan en alta carga, mientras que las celdas 04–07 son más estables. Esto puede indicar una estrategia operativa deliberada o un punto de atención para mantenimiento.

Herramientas utilizadas
Python· pandas· numpy· scikit-learn (KMeans)·matplotlib

Archivos del proyecto
📁 proyecto-flotacion-hierro/
├── limpieza_flotacion.py       # Pipeline completo de limpieza (documentado)
├── flotacion_limpio_horario.csv  # Dataset horario limpio (1,223 filas)
├── flotacion_listo_modelo.csv    # Dataset con features de lag (1,221 filas)
└── README.md                   # Este archivo
Próximo paso
Con el conjunto de datos limpio, el siguiente análisis es entrenar un modelo XGBoost para predecir el % de hierro en el concentrado y determinar qué variables tienen mayor importancia real (característica importante). Eso permitiría a los operadores enfocarse en las 3–4 palancas que realmente mueven la aguja.

Martín Jesús Villón Urday · Portafolio de Análisis de Datos · 2025 Administración de Empresas — con especialización autodidacta en análisis de datos
