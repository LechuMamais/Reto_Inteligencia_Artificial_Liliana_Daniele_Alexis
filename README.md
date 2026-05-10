# Detección de Audio Generado por IA (Deepfake Audio) - Reto Telefónica

Este proyecto se desarrolla en el marco del **Máster en Data Analytics for Business** del a **Universidad Pompeu Fabra** como respuesta al reto planteado por **Telefónica**. El objetivo principal es diseñar, entrenar y validar un modelo de aprendizaje capaz de distinguir entre grabaciones de audio reales y aquellas generadas mediante técnicas de Inteligencia Artificial (Deepfakes). Nos centraremos en buscar un modelo con buenos resultados y velocidad de ejecución.

## 1. Contexto del Proyecto

En el panorama actual de la ciberseguridad, los ataques basados en la clonación de voz y la generación de audio sintético representan una amenaza creciente. Este proyecto aborda la problemática desde una perspectiva de analítica avanzada de datos, buscando patrones imperceptibles al oído humano que permitan clasificar la autenticidad de un archivo sonoro.

## 2. Descripción del Reto

El reto consiste en una clasificación binaria:
- **Clase 0 (Real):** Audios capturados de fuentes humanas naturales.
- **Clase 1 (Fake):** Audios generados mediante arquitecturas como TTS (Text-to-Speech) o VC (Voice Conversion).

## 3. Estructura del Repositorio

- `data/`: Directorio (ignorado por git) destinado a los archivos de audio crudos y procesados.
- `Obtencion_Metricas`: Directorio con ETLs y .csv
- `Modelos/`: Notebooks Python con los modelos entrenados.
- `Metricas` : Resultados de ETL V2
- `test_voces` : Audios de voces reales a teastear
- `Conclusiones` : Conclusiones y demo

## 4. Instalación y Requisitos

Para ejecutar este proyecto, es necesario contar con Python 3.8+ y las dependencias de procesamiento de señal de audio.

1. Clonar el repositorio:
   git clone https://github.com/tu-usuario/nombre-del-repo.git

2. Crear un entorno virtual:
   python -m venv env
   source env/bin/activate  # En Windows: env\Scripts\activate

3. Instalar dependencias:

## 5. Integrantes del Equipo

Este proyecto ha sido realizado por:

- **Daniele Giovannini**
- **Liliana Espinoza Proa**
- **Alexis Mamais Mazzucco**
