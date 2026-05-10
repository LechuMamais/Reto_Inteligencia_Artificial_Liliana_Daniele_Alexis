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

