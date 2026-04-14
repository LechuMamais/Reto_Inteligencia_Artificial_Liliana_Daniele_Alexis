# Detección de Audio Generado por IA (Deepfake Audio) - Reto Telefónica

Este proyecto se desarrolla en el marco del **Máster en Data Analytics** como respuesta al reto planteado por **Telefónica**. El objetivo principal es diseñar, entrenar y validar un modelo de aprendizaje profundo (Deep Learning) capaz de distinguir entre grabaciones de audio reales y aquellas generadas mediante técnicas de Inteligencia Artificial (Deepfakes).

## 1. Contexto del Proyecto

En el panorama actual de la ciberseguridad, los ataques basados en la clonación de voz y la generación de audio sintético representan una amenaza creciente. Este proyecto aborda la problemática desde una perspectiva de analítica avanzada de datos, buscando patrones imperceptibles al oído humano que permitan clasificar la autenticidad de un archivo sonoro.

## 2. Descripción del Reto

El reto consiste en una clasificación binaria:
- **Clase 0 (Real):** Audios capturados de fuentes humanas naturales.
- **Clase 1 (Fake):** Audios generados mediante arquitecturas como TTS (Text-to-Speech) o VC (Voice Conversion).

### Metodología de Análisis
1. **Extracción de Características:** Transformación de señales de audio (WAV) en representaciones espectrales como Mel-Spectrograms o MFCCs (Mel-Frequency Cepstral Coefficients).
2. **Modelado:** Implementación de redes neuronales convolucionales (CNN) o arquitecturas basadas en Transformers para la clasificación de secuencias.
3. **Evaluación:** Uso de métricas como precisión (Accuracy), F1-Score y, especialmente, el EER (Equal Error Rate) para medir la robustez del detector.

## 3. Estructura del Repositorio

- `data/`: Directorio (ignorado por git) destinado a los archivos de audio crudos y procesados.
- `notebooks/`: Jupyter Notebooks con el análisis exploratorio (EDA) y experimentación inicial.
- `src/`: Scripts de Python para el preprocesamiento, entrenamiento del modelo y evaluación.
- `models/`: Archivos `.h5` o `.pkl` de los modelos entrenados.
- `requirements.txt`: Lista de librerías necesarias para replicar el entorno.

## 4. Instalación y Requisitos

Para ejecutar este proyecto, es necesario contar con Python 3.8+ y las dependencias de procesamiento de señal de audio.

1. Clonar el repositorio:
   git clone https://github.com/tu-usuario/nombre-del-repo.git

2. Crear un entorno virtual:
   python -m venv env
   source env/bin/activate  # En Windows: env\Scripts\activate

3. Instalar dependencias:
   pip install -r requirements.txt

## 5. Ejecución

Para entrenar el modelo con los parámetros por defecto, ejecuta:

python src/train.py --data_path ./data/train --epochs 20

Para realizar inferencia sobre un archivo nuevo:

python src/predict.py --file_path ./ruta/audio_test.wav

## 6. Integrantes del Equipo

Este proyecto ha sido realizado por los siguientes alumnos del Máster en Data Analytics:

- **Nombre Apellido** - [Enlace a GitHub/LinkedIn]
- **Nombre Apellido** - [Enlace a GitHub/LinkedIn]
- **Nombre Apellido** - [Enlace a GitHub/LinkedIn]

## 7. Agradecimientos

Agradecemos a **Telefónica** por el planteamiento del reto y por facilitar los datasets necesarios para la investigación en materia de seguridad y detección de fraude por IA.