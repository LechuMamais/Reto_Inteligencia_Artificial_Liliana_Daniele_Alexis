# PASOS Y DECISIONES DEL PROYECTO

## Contexto inicial

Este proyecto nace con un objetivo claro: detectar audios falsos (spoof) frente a audios reales (bonafide) a partir de métricas acústicas. Desde el inicio, el reto no fue solo entrenar un modelo, sino decidir **cómo representar el audio** para que el algoritmo aprendiera señales realmente útiles.

El primer trabajo fue entender bien el dataset y su protocolo: qué significaba cada fichero, cómo leer sus etiquetas (`bonafide`/`spoof`) y cómo alinear audio + metadata para construir una base de entrenamiento consistente.

## Primera aproximación: ETL V1 (agregación por audio)

En la primera versión de extracción de métricas (`Obtencion_Metricas/ETL_V1.ipynb`) seguimos una estrategia clásica: extraer muchas métricas por ventanas y, para consolidar una sola fila por archivo, calcular:

- media de cada métrica
- desviación estándar de cada métrica

La decisión tenía sentido para arrancar rápido:

- permitía transformar cientos de ventanas en un solo registro por audio
- reducía complejidad de modelado en la primera fase
- hacía viable entrenar modelos tabulares de manera simple

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