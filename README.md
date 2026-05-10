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
├── Obtencion_Metricas/            # ETL Parte 1: extracción de features estadísticas
│   ├── ETL_V1.ipynb               # ETL tabular (MFCCs, ZCR, RMSE...) — dataset train
│   └── ETL_V1_test.ipynb          # ETL tabular — dataset eval
├── Metricas/                      # ETL Parte 2: tensores STFT para CNN
│   ├── ETL_V2.1_train/            # Tensores ASVspoof train (A01-A06)
│   ├── ETL_V2.1_eval/             # Tensores ASVspoof eval (A07-A19)
│   └── ETL_LatinAmerica/          # Tensores dataset Latinoamérica
├── Modelos/
│   └── Modelos_ETL1/              # Parte 1: EDA y modelos Random Forest, XGBoost, MLP Red Neuronal
│       ├── Modelo_EDA_ETL1.ipynb
│       ├── Modelo1_ETL1.ipynb
│       ├── Modelo2_ETL1.ipynb
│       ├── Modelo3_ETL1.ipynb
│       └── Modelo4_ETL1.ipynb
├── Modelos_CNN/                   # Parte 2: modelos CNN
│   ├── Modelo2.ipynb
│   ├── Modelo2_1.ipynb
│   ├── Modelo2_1_Balanceado.ipynb
│   ├── Modelo3_Latinoamerica.ipynb
│   └── (otros notebooks de comparativa y testing)
├── test_voces/                    # Grabaciones propias para evaluación en condiciones reales
├── Graficos/                      # Gráficos generados por los modelos
└── Conclusiones/                  # Notas y conclusiones finales
```

> Los archivos de audio y los tensores `.npy` de gran tamaño están excluidos del repositorio por `.gitignore`.

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

- Daniele Giovannini
- Liliana Espinoza Proa
- Alexis Mamais Mazzucco

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


# PARTE 2: ETL 2 e implementación de modelos