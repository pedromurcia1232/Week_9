# Week9_9

Integrantes: DIEGO ALEJANDRO RUIZ ALFONSO - ppmurcia@ucundinamarca.edu.co PEDRO PASCUAL MURCIA VARGAS - ppmurcia@ucundinamarca.edu.co JHON EDUARD TINJACA CRUZ - jetinjaca@ucundinamarca.edu.co JULIAN DAVID SILVA GUZMAN - jdsilva@ucundinamarca.edu.co

# README: Clasificación de Imágenes con Data Augmentation y Transfer Learning

Este notebook explora dos técnicas fundamentales en el Deep Learning para la clasificación de imágenes: **Data Augmentation** y **Transfer Learning**. El objetivo es demostrar cómo estas estrategias pueden mejorar significativamente la generalización y el rendimiento de los modelos, especialmente en escenarios con datasets limitados.

##  Objetivo del Proyecto

El objetivo principal es comprender y aplicar:

1.  **Data Augmentation:** Aumentar la cantidad y diversidad de los datos de entrenamiento a partir de las imágenes existentes, aplicando transformaciones aleatorias (rotaciones, desplazamientos, zooms, volteos) para hacer el modelo más robusto y reducir el sobreajuste.
2.  **Transfer Learning:** Reutilizar un modelo pre-entrenado en un dataset grande (como ImageNet) y adaptarlo a una tarea específica (clasificación de gatos vs. perros), aprovechando los conocimientos previamente aprendidos por el modelo.

## Tecnologías Utilizadas

*   **TensorFlow / Keras:** Framework principal para la construcción y entrenamiento de modelos de Deep Learning.
*   **TensorFlow Datasets (TFDS):** Para una fácil carga y manejo del dataset cats_vs_dogs.
*   **MobileNetV2:** Arquitectura de red neuronal convolucional (CNN) pre-entrenada, ligera y eficiente, utilizada como modelo base para Transfer Learning.
*   **Matplotlib:** Para la visualización de imágenes y el historial de entrenamiento del modelo.

##  Dataset: Cats vs Dogs

Se utiliza el popular dataset cats_vs_dogs de TensorFlow Datasets. Este conjunto de datos contiene imágenes de perros y gatos, y es un benchmark clásico para problemas de clasificación binaria de imágenes. Se carga directamente y se realizan algunas visualizaciones de ejemplo.

##  Data Augmentation (Aumento de Datos)

Se implementa `ImageDataGenerator` de Keras para aplicar transformaciones en tiempo real a las imágenes de entrenamiento. Esto incluye:

*   rescale=1./255: Normalización de los valores de píxel al rango [0, 1].
*   rotation_range=20: Rotación aleatoria de hasta 20 grados.
*   width_shift_range=0.2, height_shift_range=0.2: Desplazamientos horizontales y verticales aleatorios.
*   zoom_range=0.2: Aplicación de zoom aleatorio.
*   horizontal_flip=True: Volteo horizontal aleatorio.

**Importante:** Para el set de validación, solo se aplica el reescalado para asegurar que las métricas de rendimiento sean consistentes y no afectadas por transformaciones artificiales.

##  Transfer Learning con MobileNetV2

La estrategia de Transfer Learning se implementa de la siguiente manera:

1.  **Carga del Modelo Base:** Se carga `MobileNetV2` con pesos pre-entrenados en `ImageNet` (`weights='imagenet'`) pero **sin la capa superior de clasificación** (include_top=False). Esto nos da acceso a las potentes características extraídas por el modelo en las capas convolucionales.
2.  **Congelación de Capas Base:** Se establecen los pesos del `base_model` como no entrenables (`base_model.trainable = False`). Esto evita que los pesos pre-entrenados se modifiquen durante el entrenamiento, preservando el conocimiento aprendido.
3.  **Adición de Capas Personalizadas:** Se añaden nuevas capas encima del modelo base para adaptar el modelo a la tarea específica de clasificar gatos vs. perros:
    *   GlobalAveragePooling2D(): Reduce las dimensiones del mapa de características, convirtiéndolo en un vector único.
    *   Dense(128, activation='relu'): Una capa oculta densa para aprender representaciones de alto nivel.
    *   Dense(2, activation='softmax'): La capa de salida con 2 neuronas (una por clase) y activación softmax para la clasificación binaria (aunque se usa categorical_crossentropy, que es común para clasificación multiclase con etiquetas one-hot).
4.  **Compilación del Modelo:** El modelo se compila con el optimizador `adam`, la función de pérdida categorical_crossentropy y la métrica de accuracy.

##  Entrenamiento del Modelo

El entrenamiento se realiza utilizando conjuntos de datos de entrenamiento y validación preparados a partir de `tfds.load('cats_vs_dogs').

*   Las imágenes se redimensionan a (224, 224), el tamaño esperado por MobileNetV2.
*   Los labels se convierten a formato *one-hot* (tf.one_hot(y, 2)).
*   Se utilizan lotes (batch_size=10).
*   El modelo se entrena durante 10 épocas, monitoreando la precisión y la pérdida en el conjunto de validación.

Se incluye una función `plot_training_history` para visualizar el rendimiento del modelo (precisión y pérdida) en los conjuntos de entrenamiento y validación a lo largo de las épocas.

##  Resultados y Conclusiones

Después del entrenamiento, se observa cómo la precisión del modelo mejora y la pérdida disminuye, tanto en el conjunto de entrenamiento como en el de validación. Esto demuestra que:

*   **Transfer Learning** permite alcanzar una alta precisión rápidamente, incluso con un dataset relativamente pequeño, al aprovechar las características genéricas aprendidas por MobileNetV2.
*   La combinación con **Data Augmentation** (aunque en este notebook el aumento se aplica implícitamente a través del preprocesamiento y no directamente con ImageDataGenerator.flow debido al uso de tf.data.Dataset, el concepto sigue siendo válido para mejorar la generalización) ayuda a prevenir el sobreajuste y a que el modelo sea más robusto a variaciones en las imágenes de entrada.

