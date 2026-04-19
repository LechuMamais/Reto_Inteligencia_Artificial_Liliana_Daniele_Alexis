1. La carpeta clave: ASVspoof2019_LA_cm_protocols

"CM" significa Countermeasure (Medida de Seguridad o Contramedida). Esta es la carpeta que realmente te interesa para tu objetivo actual. Aquí están los metadatos y las etiquetas para entrenar tu modelo detector de deepfakes.

    * ASVspoof2019.LA.cm.train.trn.txt: Este es tu archivo principal. Contiene las etiquetas (bona fide o spoof) y la información del tipo de ataque para todos los audios de la carpeta de entrenamiento (train). Es el archivo del que te hablé en mi respuesta anterior y el que debe leer tu script de Python.

    1ª Columna: ID del Locutor (LA_0079 / LA_0089)

        Es el identificador único de la persona (el hablante base). En el primer caso, es el locutor 79; en el segundo, el locutor 89. Esto es útil para asegurarte de que tu modelo no está aprendiendo simplemente a reconocer la voz de una persona específica, sino a detectar los artefactos de la IA sin importar quién hable.

    2ª Columna: Nombre del Archivo (LA_T_1138215 / LA_T_2947816)

        Es el nombre exacto del archivo de audio correspondiente, pero sin la extensión. Para cargar el audio en tu script de Python, tendrás que concatenarle .flac al final (por ejemplo: LA_T_1138215.flac).

    3ª Columna: Entorno / Sistema (-)

        En la partición LA (Logical Access), esta columna siempre es un guion (-). Se deja vacía por compatibilidad de formato con la otra partición del reto (PA - Physical Access), donde esta columna se usa para indicar el tamaño de la habitación o la distancia del micrófono. Como aquí evaluamos ataques por software/red, no aplica.

    4ª Columna: ID del Ataque (- / A05)

        Indica qué tecnología se usó para generar el audio.

        Si es un guion (-), significa que no hay ataque: es un audio de voz 100% humana y original.

        Si tiene un código (como A05), indica el algoritmo específico de IA (Text-to-Speech o Voice Conversion) que generó el deepfake.

    5ª Columna: La Etiqueta (bonafide / spoof)

        Esta es tu variable objetivo (target) para entrenar el modelo.

        bonafide (de buena fe): Voz real.

        spoof (falsificación): Voz generada por IA.

    * ASVspoof2019.LA.cm.dev.trl.txt: Es el protocolo para tu conjunto de validación (development/dev). Lo usarás para afinar los hiperparámetros de tu modelo sin tocar los datos de test.

    * ASVspoof2019.LA.cm.eval.trl.txt: Es el protocolo para el conjunto de evaluación final (eval). Este es el que contiene los ataques desconocidos para poner a prueba la generalización de tu modelo.

2. La carpeta biométrica: ASVspoof2019_LA_asv_protocols (No nos interesa)

"ASV" significa Automatic Speaker Verification (Verificación Automática de Locutor).

El ASVspoof asume que el detector de deepfakes se va a usar para proteger un sistema biométrico (por ejemplo, el acceso por voz al banco). Estos archivos contienen listas de qué audios deben compararse con qué locutores para saber si el sistema biométrico acierta o falla. No los necesitas para entrenar un clasificador de "real vs fake".

Si te da curiosidad la nomenclatura de los archivos dentro:

    trn (Enrolment): Listas de audios reales usados para registrar o "matricular" la voz de un usuario en el sistema.

    trl (Trials): Listas de intentos de acceso (algunos del usuario real, otros de impostores, otros generados por IA) para probar el sistema.

    female / male / gi: Los protocolos están separados para evaluar el sistema en mujeres, hombres, o de forma independiente del género (gi = gender independent).

3. La carpeta de evaluación: ASVspoof2019_LA_asv_scores (No nos interesa)

Para evaluar los modelos en la competición oficial, los organizadores usaron una métrica combinada llamada t-DCF (tandem Detection Cost Function). Esta métrica castiga a tu modelo de forma diferente dependiendo de si tu error de detección provoca que un sistema de reconocimiento de voz deje pasar a un impostor o bloquee a un usuario legítimo.

    *.scores.txt: Para que los participantes no tuvieran que construir su propio sistema biométrico desde cero para calcular esta métrica, los organizadores incluyeron las puntuaciones (scores) precalculadas de un sistema de reconocimiento de voz base (baseline). Se usan estrictamente junto con un script oficial para calcular el t-DCF final.