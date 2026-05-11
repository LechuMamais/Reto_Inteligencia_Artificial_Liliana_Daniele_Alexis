# Detección de Audio Generado por IA (Deepfake Audio) - Reto Telefónica

Este proyecto se desarrolla en el marco del **Máster en Data Analytics for Business (UPF BSM)** como respuesta al reto planteado por **Telefónica**. El objetivo es diseñar, entrenar y validar modelos capaces de distinguir entre grabaciones de voz humana real y voz generada mediante Inteligencia Artificial (deepfake audio).

## 1. Contexto del Proyecto

Este proyecto aborda la problemática desde una perspectiva de analítica avanzada de datos, buscando patrones acústicos imperceptibles al oído humano que permitan clasificar la autenticidad de un archivo sonoro.

## 2. Descripción del Reto

El reto consiste en una **clasificación binaria**:
- **Clase 0 - Bonafide:** audios capturados de fuentes humanas naturales.
- **Clase 1 - Spoof:** audios generados mediante arquitecturas de síntesis de voz (TTS / Voice Conversion).

### Metodología de Análisis

El proyecto se divide en dos enfoques complementarios implementados en paralelo:

- Parte 1: ETL clásico + Machine Learning: extracción de métricas acústicas estadísticas (MFCCs, ZCR, RMSE, espectro) para generar un dataset tabular y entrenar modelos Random Forest, XGBoost y Red Neuronal MLP con validaciones progresivamente más exigentes.

- Parte 2: Deep Learning con CNN: transformación de los archivos de audio en espectrogramas mediante STFT y entrenamiento de Redes Neuronales Convolucionales (CNN) que tratan cada espectrograma como una imagen acústica.

### Métricas de evaluación
F1-Score y matriz de confusión (Recall y Precision) como métricas principales.

## 3. Dataset

- ASVspoof 2019 - Logical Access (LA)

| Partición | Audios | Descripción |
|-----------|--------|-------------|
| Train | 25.380 | Ataques A01–A06 + bonafide |
| Dev | 24.844 | Ataques A01–A06 + bonafide |
| Eval | 71.237 | Ataques A07–A19 + bonafide |

- FinalDataset_16khz (Latinoamérica) - usado en Parte 2

| Partición | Audios | Descripción |
|-----------|--------|-------------|
| Total | 80.816 | CycleGAN, Diff, StarGAN, TTS + bonafide en español |

## 4. Estructura del Repositorio

```
├── Obtencion_Metricas/                         # ETL Parte 1: extracción de features estadísticas
│   ├── ETL_V1.ipynb                            # ETL tabular (MFCCs, ZCR, RMSE...) - dataset train
│   ├── ETL_V1_test.ipynb                       # ETL tabular - dataset eval
│   ├── ETL_V2.ipynb                            # ETL Parte 2: generación de espectrogramas STFT
│   ├── train_EDA_Daniele.csv                   # Dataset train balanceado (features acústicas)
│   ├── test_EDA_Daniele.csv                    # Dataset test (features acústicas)
│   ├── dataset_balanceado_5000_EDA_Daniele.csv # Muestra balanceada 50/50 de 5.000 audios
│   ├── dataset_caracteristicas_eval.csv        # Features del conjunto eval (A07-A19)
│   └── feature_ranking_cohen_d_Daniele.csv     # Ranking de features por d de Cohen
├── Metricas/                                   # Tensores STFT (.npy) - excluidos por .gitignore
├── Modelos/
│   ├── Modelos_ETL1/                           # Parte 1: EDA y modelos Random Forest, XGBoost, MLP
│   │   ├── Modelo_EDA_ETL1.ipynb               # Análisis exploratorio y undersampling estratificado
│   │   ├── Modelo1_ETL1.ipynb                  # XGBoost optimizado con 34 features
│   │   ├── Modelo2_ETL1.ipynb                  # Reducción de features por Permutation Importance
│   │   ├── Modelo3_ETL1.ipynb                  # Validación estructural por tipo de ataque
│   │   ├── Modelo4_ETL1.ipynb                  # Leave-One-Group-Out Cross Validation (LOGOCV)
│   │   └── config_modelo_final.json            # Configuración del modelo final serializado
│   ├── Modelos_ETL2/
│   │   ├── Generación/                         # Generación y comparativa de arquitecturas CNN
│   │   │   ├── SpatialCNN_2x4_Flatten.ipynb    # Primera CNN espacial (pooling 2x4, cabeza Flatten)
│   │   │   ├── SpatialCNN_1x8_Flatten.ipynb    # CNN espacial con pooling 1x8
│   │   │   ├── SpatialCNN_1x8_Flatten_CrossValid.ipynb  # LOAO cross-validation por tipo de ataque
│   │   │   ├── ETL2_Comparativa_Flatten_vs_GAP.ipynb    # Modelo 2.3: pooling vs cabeza GAP
│   │   │   └── ETL2_Comparativa_Arquitecturas.ipynb     # Modelo 2.2: comparativa A/B/C arquitecturas
│   │   └── Analisis_&_Comparativa/             # Análisis de resultados, testing y ETL adicional
│   │       ├── Modelo2.1_ETL2.ipynb
│   │       ├── Modelo3_Testing_Voces_Equipo.ipynb
│   │       └── ETL_V3.ipynb
│   ├── Modelo2.1_ETL2.ipynb                    # CNN SpatialCNN - entrenamiento principal
│   ├── Modelo2.1_ETL2_ASVyLatin.ipynb          # CNN entrenada con ASVspoof + Latinoamérica
│   ├── Modelo2.1_ETL2_Testing_Voces_Equipo.ipynb
│   ├── Modelo2_ETL2_TestingLiliana.ipynb
│   ├── Modelo2.1_Alexis.ipynb
│   ├── Modelo2.2_Alexis.ipynb
│   ├── Modelo 2.2_Alexis_Comparativa.ipynb
│   ├── Modelo3_Testing_Voces_Equipo.ipynb
│   └── ETL_V3.ipynb                            # ETL adicional para nuevos datasets
├── Graficos/                                   # Gráficos generados por los modelos
│   ├── Modelo1_ETL1/
│   ├── Modelo2_ETL1/
│   ├── Modelo3_ETL1/
│   ├── Modelo4_ETL1/
│   ├── ModeloEDA_ETL1/
│   └── EDA_ETLV2/
├── test_voces/                                 # Grabaciones reales del equipo (Daniele, Liliana, Alexis)
│   └── (.m4a / .wav / .mp3 — excluidos por .gitignore)
├── Conclusiones/                               # Análisis de errores y demo del sistema
│   ├── Demo.ipynb
│   └── Razonamiento_Errores_Modelo.ipynb
├── README_PARTE1.md
├── README_PARTE2.md
└── Notas sobre el dataset.md
```

> Los archivos de audio (`.m4a`, `.wav`, `.mp3`, `.flac`) y los tensores `.npy` de gran tamaño están excluidos del repositorio por `.gitignore`.

## 5. Instalación y Requisitos

Python con las siguientes dependencias principales:

librosa
numpy
pandas
scikit-learn
xgboost
tensorflow
matplotlib
seaborn
joblib
tqdm
ffmpeg  # para conversión de audio en tests de voces reales

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## 6. Integrantes del Equipo

Este proyecto ha sido realizado por:

- **Daniele Giovannini**
- **Liliana Espinoza Proa**
- **Alexis Mamais Mazzucco**

## 7. Agradecimientos

Agradecemos a **Telefónica** por el planteamiento del reto y por facilitar los datasets necesarios para la investigación en materia de seguridad y detección de fraude por IA.


# PARTE 1: ETL 1 e implementación de modelos

## ETL (ETL_V1.ipynb / ETL_V1_test.ipynb)

A partir de los archivos `.flac` del conjunto de entrenamiento (25.380 audios) y del conjunto de evaluación (71.237 audios), extrajimos métricas acústicas con **librosa** para generar un dataset tabular listo para el entrenamiento de modelos de Machine Learning.

Por cada audio se calcularon estadísticas de media y desviación estándar sobre ventanas temporales de 20–40 ms, obteniendo **34 features por audio** (más la columna de 'file name', 'attack_ID' y 'label'):

- **Señal:** media y std de la forma de onda, RMSE, Zero-Crossing Rate, Tempo;
- **MFCCs:** media y std de los 13 coeficientes Mel-Frequency Cepstral (µ y σ);
- **Espectral:** centroide, bandwidth y rolloff espectrales.

El resultado es un CSV con una fila por audio, 34 columnas de features, la etiqueta `label` (bonafide/spoof) y el identificador del algoritmo de ataque (`attack_id`).

Se descartaron intencionalmente el Tempogram, las Polyfeatures y el Pitch fundamental por su coste computacional elevado y su baja relevancia en detección de voz sintética.


## EDA (Modelo_EDA_ETL1.ipynb)

El análisis exploratorio se realizó exclusivamente sobre el conjunto de train, con las siguientes decisiones clave:

- **Undersampling estratificado por attack_id:** el dataset original tiene un desbalanceo extremo (22.800 spoof vs 2.580 bonafide). Se optó por undersampling estratificado en lugar de oversampling (SMOTE) para evitar generar vectores acústicos sintéticos que no corresponden a patrones reales. Resultado: **5.000 audios balanceados al 50/50**, con todos los tipos de ataque representados proporcionalmente.

- **Separación Train/Test antes del EDA:** la partición 80/20 (4.000 train / 1.000 test) se realizó antes de cualquier análisis para evitar Data Leakage. El test quedó sellado hasta la evaluación final.

- **Outliers conservados:** en datos de audio, los valores extremos son acústicamente plausibles y contienen información discriminativa relevante.

- **d de Cohen:** se midió el poder discriminativo de cada feature entre clases. Las features con mayor capacidad de separación fueron: `mfcc_4_std`, `spectral_bandwidth_mean`, `mfcc_1_std`, `mfcc_8_mean`, `mfcc_6_mean`.


## Modelo 1: XGBoost optimizado con 34 features (Modelo1_ETL1.ipynb)

Se compararon tres algoritmos (Random Forest, XGBoost, Red Neuronal MLP) optimizando sus hiperparámetros con **RandomizedSearchCV** y evaluando con **validación cruzada estratificada de 5 folds** sobre los 4.000 audios de train.

Se eligió **XGBoost** por su mejor balance entre rendimiento tabular y capacidad de generalización con datasets de tamaño pequeño-mediano, frente al MLP que necesita volúmenes de datos mayores para converger.

| Métrica | CV Val | Test | AUC |
|---------|--------|------|-----|
| F1-Score | 0.9484 | 0.9575 | 0.9910 |


## Modelo 2: Reducción de features por Permutation Importance (Modelo2_ETL1.ipynb)

Se aplicó **Permutation Importance** sobre el XGBoost optimizado para identificar las features que realmente contribuyen al F1-score. De las 34 features originales, **13 resultaron con importancia positiva** (reducción del 62%). Dominan `signal_mean` y los coeficientes MFCC (media y std).

| Modelo | Features | CV Val F1 | Test F1 | Test AUC |
|--------|----------|-----------|---------|----------|
| M1 (34 features) | 34 | 0.9484 | 0.9575 | 0.9910 |
| M2 (13 features) | 13 | 0.9093 | 0.9032 | 0.9700 |

El trade-off (−5% F1 a cambio de −62% features) es favorable en sistemas de detección en tiempo real, donde calcular 13 coeficientes en lugar de 34 reduce directamente el coste de respuesta.


## Modelo 3: Validación estructural por ataque (Modelo3_ETL1.ipynb)

Se diseñó una **separación estructural por tipo de ataque**: entrenamiento con A01–A04, test con A05–A06 nunca vistos. Esto simula el escenario real donde aparecen nuevos sistemas TTS en producción.

| Fase | F1 | AUC |
|------|----|-----|
| CV sobre A01–A04 | 0.9445 | 0.9874 |
| Test A05–A06 (nunca vistos) | 0.5021 | 0.7665 |

La caída confirma que el modelo sobreajustó los patrones acústicos de A01–A04. Evaluando por ataque individual: A05 (F1=0.6918) vs A06 (F1=0.1948), revelando que A06 tiene una arquitectura de síntesis acústicamente muy diferente.


## Modelo 4 — Leave-One-Group-Out Cross Validation (Modelo4_ETL1.ipynb)

Se implementó **LOGOCV** (6 folds, cada ataque excluido del train una vez) para obtener una estimación estadísticamente sólida de la capacidad de generalización.

| Ataque excluido | F1-Score |
|-----------------|----------|
| A01 | 0.9409 |
| A02 | 0.9501 |
| A03 | 0.6051 |
| A04 | 0.7060 |
| A05 | 0.8056 |
| A06 | 0.2200 |
| **Media** | **0.7039 ± 0.27** |

La STD de 0.27 refleja que el modelo detecta bien ataques acústicamente similares a los vistos (A01, A02) pero falla ante algoritmos con vocoders radicalmente distintos (A06).

**Evaluación sobre conjunto eval (A07–A19, 71.237 audios):** weighted avg F1 = 0.84, con bajo rendimiento en bonafide (F1=0.47) por el desbalanceo extremo del conjunto (90% spoof).

**Evaluación sobre grabaciones propias (test_voces):** de 31 grabaciones reales hechas con dispositivos heterogéneos (iPhone, WhatsApp, ordenadores), solo 10 se clasificaron correctamente como bonafide. El mismatch de dominio entre el entorno controlado del ASVspoof 2019 y grabaciones reales con ruido ambiental explica el bajo rendimiento.

## Límite detectado en V1

Al avanzar, vimos el principal problema de este enfoque: al resumir todo el audio con media y desvío, se perdía información local importante.

Esto afectaba directamente al problema de negocio:

- un audio puede ser mayormente real y tener solo un tramo manipulado
- al promediar todo, ese tramo puede quedar diluido
- además, V1 obliga a procesar el archivo completo antes de decidir, lo que puede alargar detección

En otras palabras, V1 fue una buena base para arrancar, pero no capturaba bien eventos acústicos localizados.

## Giro estratégico: ETL V2

A partir de ese hallazgo, cambiamos el enfoque en `Obtencion_Metricas/ETL_V2.ipynb`: pasar de una representación global resumida a una representación local estructurada.

### Decisión técnica central

Para cada archivo de audio:

- seleccionar 5 muestras (frames)
- extraer en cada muestra 1025 métricas espectrales de Fourier
- construir una matriz por audio de forma `(5, 1025)`

Con todos los archivos juntos, la variable de entrada pasa a ser un dataset 3D (`.npy`), preservando estructura temporal/frecuencial mucho más rica que en V1.

### Decisiones de robustez del pipeline

En ETL V2 también tomamos decisiones de ingeniería para evitar sesgos y ganar consistencia:

- **Sample rate unificado a 16 kHz** para mantener coherencia temporal entre audios de origen distinto.
- **Normalización de amplitud** para evitar que el modelo aprenda volumen en lugar de patrones acústicos de spoofing.
- **Pipeline tolerante a errores** (archivos corruptos/faltantes) y procesamiento escalable para no frenar la ETL completa.
- **Etiquetado binario consistente** (`0=bonafide`, `1=spoof`) para compatibilidad directa con entrenamiento.

## Etapa de Modelos CNN (Parte 2): iteración y contraste de hipótesis

Con ETL V2 empezamos a iterar diferentes modelos (`Modelos/Modelos_ETL2/Generación/SpatialCNN_2x4_Flatten.ipynb`, `Modelos/Modelos_ETL2/Generación/ETL2_Comparativa_Flatten_vs_GAP.ipynb`, `Modelos/Modelos_ETL2/Generación/ETL2_Comparativa_Arquitecturas.ipynb`).

La lógica no fue buscar una única arquitectura "mágica", sino comparar hipótesis y aprender de cada comportamiento.

## Por qué usamos modelos convolucionales

La decisión de usar CNN no fue solo por tendencia de literatura, sino por cómo se veía la información en el análisis gráfico de frecuencias:

- al representar las 1025 bandas por frame, aparecían patrones locales de energía (zonas vecinas en frecuencia que suben/bajan juntas)
- los artefactos de spoofing no suelen ser un único pico aislado, sino estructuras locales repetibles en regiones del espectro
- con 5 frames por audio, el input tiene forma de "imagen acústica" (`tiempo x frecuencia`)

Las CNN encajan bien en ese escenario porque:

- aprovechan **correlación local** mediante filtros pequeños
- comparten pesos (menos parámetros que una red densa equivalente)
- tienden a generalizar mejor cuando hay estructura espacial/frecuencial

Por eso, en lugar de aplanar los datos y perder topología acústica, priorizamos arquitecturas convolucionales.

## Las 3 arquitecturas comparadas (definición + validez)

En `Modelos/Modelos_ETL2/Generación/ETL2_Comparativa_Arquitecturas.ipynb` cerramos la comparativa con tres hipótesis complementarias:

Antes de cerrar esa versión, también fuimos afinando **hiperparámetros estructurales de CNN** (no solo capas densas o regularización), especialmente:

- tamaño de kernel convolucional
- tipo y tamaño de pooling
- pooling asimétrico para no colapsar el eje temporal corto (5 frames)

### A) `A_spatial_cnn` (mezcla local tiempo-frecuencia)

**Definición:** CNN 2D que aplica kernels sobre ambos ejes (`tiempo` y `frecuencia`) para capturar patrones conjuntos.

**Configuración que terminamos comparando:** kernels `3x3` + `MaxPooling2D(1,8)` por bloque.

**Por qué es válida:**

- modela explícitamente interacción entre dinámica temporal corta y estructura espectral
- si el spoof deja huellas que dependen del contexto local tiempo-frecuencia, esta arquitectura debería capturarlas mejor
- mantiene una inductive bias adecuada para datos tipo espectrograma

### B) `B_freq_only_cnn` (convolución solo frecuencial)

**Definición:** CNN 2D restringida al eje de frecuencia (kernels tipo `(1, k)`), minimizando mezcla temporal local.

**Configuración que terminamos comparando:** kernels `(1,9)` y `(1,5)` con `MaxPooling2D(1,4)` + `GlobalAveragePooling2D`.

**Por qué es válida:**

- permite aislar si la mayor señal discriminante está en la forma del espectro y no en la transición temporal
- funciona como experimento controlado frente a la opción A
- es más liviana computacionalmente y útil como baseline eficiente

### C) `C_independent_frames` (encoder por frame + agregación global)

**Definición:** procesa cada frame por separado (bloques compartidos) y luego agrega la información global para clasificar.

**Configuración que terminamos comparando:** `Conv1D` por frame (kernels `9` y `5`) con `MaxPooling1D(4)` y agregación final (`GlobalMaxPooling1D` + `GlobalAveragePooling1D`).

**Por qué es válida:**

- es coherente con ETL V2 cuando los 5 frames son equidistantes y no necesariamente contiguos
- evita suponer continuidad temporal fuerte entre muestras separadas
- permite capturar calidad por frame y fusionarla al final de forma robusta

## Resultados comparativos en diferentes conjuntos

La comparación se hizo con protocolo homogéneo para que fuera justa:

- mismo `train`
- validación común en `dev`
- calibración de umbral por modelo en `dev` maximizando F1 de `spoof`
- evaluación final en `eval` sin recalibrar (para medir generalización real)
- evaluación en audios grabados con nuestros telefonos y computadoras

Además, se mantuvo batería de métricas robustas para desbalance (`balanced_accuracy`, `F1_spoof`, `ROC-AUC`, `PR-AUC`, matrices de confusión y métricas de error probabilístico).

### Lectura operativa de los resultados

- el ranking no fue idéntico en todos los conjuntos, lo que confirmó sensibilidad al cambio de distribución
- hubo modelos más rápidos de entrenar (por ejemplo, `B_freq_only_cnn`) y otros más costosos pero potencialmente más expresivos (`C_independent_frames`)
- la calibración de umbral en `dev` evitó comparar modelos con un 0.5 fijo que podía sesgar la interpretación
- el análisis de `FP/FN` ayudó a discutir coste real del error (ataques que se escapan vs bloqueos de audios legítimos)

Este punto fue clave para presentar resultados sin sobreoptimismo: no solo mirar "qué modelo da mayor score", sino también dónde falla y con qué coste.

## Comprobación específica por `attack_id`

Para validar robustez real frente a tipos de ataque, se incorporó una comprobación específica en `Modelos/Modelos_ETL2/Generación/SpatialCNN_1x8_Flatten_CrossValid.ipynb` basada en **Leave-One-Attack-Out (LOAO)**:

- en cada fold se deja fuera un `attack_id` completo para test
- se entrena con el resto de `attack_id` spoof
- se compara desempeño por ataque retenido (heatmaps de `balanced_accuracy` y `MCC`)

Este análisis permite responder una pregunta crítica:  
**¿el modelo aprende señales generales de spoofing o memoriza atacantes concretos?**

También dejó una lección técnica importante para el pipeline: para que esta validación sea automática y reproducible, `y_labels` debe conservar `attack_id` en formato consistente durante la exportación ETL.

## Ajuste de kernels y pooling: qué aprendimos

En la evolución de los modelos no solo comparamos "arquitecturas diferentes", también ajustamos cómo cada una comprime la información:

- en entradas `(5,1025)`, el eje temporal es muy corto y el frecuencial muy largo
- por eso priorizamos **pooling asimétrico** (`(1,8)` o `(1,4)`) para comprimir frecuencia sin destruir tiempo
- variantes de kernel más "anchas" en frecuencia (`(1,9)`) sirvieron para captar estructuras espectrales locales extendidas
- kernels `3x3` se mantuvieron como opción equilibrada cuando queríamos mezclar tiempo y frecuencia localmente

Conclusión práctica de diseño:  
el rendimiento no dependía solo de "qué arquitectura" elegíamos, sino también de **cómo definíamos kernels y pooling** en función de la geometría real del tensor acústico.

## Problemas reales encontrados y decisiones tomadas

### 1) Desbalance de clases y métricas engañosas

**Problema:** el conjunto tiene predominio de spoof respecto a bonafide, por lo que la `accuracy` por sí sola podía dar una sensación falsa de buen rendimiento.

**Decisiones:**

- dejar de evaluar solo con accuracy
- aplicar class weight y data augmentation
- incorporar métricas más informativas: AUC, F1, precision, recall, balanced accuracy
- en versiones posteriores, cuidar explícitamente la evaluación por clase objetivo (spoof)

### 2) Sobreajuste durante entrenamiento

**Problema:** en algunos ciclos de entrenamiento aparecía separación entre mejora en train y estancamiento/deterioro en validación.

**Decisiones:**

- usar `BatchNormalization` para estabilizar activaciones
- usar `Dropout` para reducir co-adaptación de neuronas
- usar `EarlyStopping` para detener cuando ya no mejora validación y restaurar mejores pesos

Resultado: entrenamientos más controlados y menos dependencia de una sola corrida afortunada.

### 3) Cambio de distribución (hallazgo clave)

**Problema:** detectamos escenarios con buena validación pero caída fuerte en test/eval, incluyendo casos con rendimiento cercano al azar en AUC. Esto confirmó que una validación interna alta no garantizaba generalización real.

**Decisiones:**

- separar claramente conjuntos y roles:
  - `dev` para ajustar y calibrar decisión
  - `eval` reservado para validación final de generalización
- evitar cerrar conclusiones solo con métricas de entrenamiento/validación interna

### 4) Duda arquitectónica real: no asumir una sola red

**Problema:** con datos `(5, 1025)`, no era obvio si convenía mezclar tiempo-frecuencia localmente, operar por frecuencia o tratar frames de forma más independiente.

**Decisión:** convertir esa duda en experimento explícito comparando hipótesis A/B/C:

- `A_spatial_cnn`: mezcla local tiempo-frecuencia (CNN 2D)
- `B_freq_only_cnn`: convolución focalizada en frecuencia
- `C_independent_frames`: procesamiento por frame + agregación posterior

Así pasamos de discutir opiniones a contrastar evidencia.

## Qué resolvimos realmente (sin ganador absoluto)

La resolución de esta etapa no es "un modelo final único", sino algo más sólido para el equipo:

- una **ETL V2** que conserva información local crítica para spoofing
- un **marco de evaluación más estricto** (dev para calibrar, eval para generalizar)
- una **metodología comparativa de arquitecturas** basada en hipótesis

Esto evita sobreprometer resultados y deja una base reproducible para siguientes iteraciones.

## Lecciones aprendidas

- Más métricas no es complejidad innecesaria: es protección contra el autoengaño experimental.
- Una validación buena no garantiza rendimiento en escenarios nuevos.
- La calidad de la ETL define el techo del modelo tanto como la arquitectura.
- Medir, comparar y documentar decisiones fue tan importante como entrenar redes.

## Próximos pasos sugeridos

- Consolidar la comparativa A/B/C con el mismo protocolo de evaluación en todas las corridas.
- Analizar con más detalle casos de error en audios reales del equipo.
- Explorar ajustes finos de muestreo en ETL V2 para robustez cross-domain.