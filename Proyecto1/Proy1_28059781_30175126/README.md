# PIMES — Predictor de Interacciones de Medicamentos y Efectos Secundarios

Proyecto 1 — *Inteligencia Artificial: Aprendizaje Automático* (UCAB, Marzo–Julio 2026).

Predice, para cada medicamento descrito por su vector de 881 subestructuras químicas, la probabilidad de que provoque cada uno de los 1385 efectos secundarios estudiados. La predicción la hace una red neuronal artificial *poco profunda* (MLP con tres capas ocultas: 512, 256 y 128 neuronas) con activación ReLU y salida sigmoide por etiqueta.

## Requisitos

- Python 3.10 o superior
- Paquetes: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`

Instalación:

```
pip install numpy pandas scikit-learn matplotlib joblib
```

## Estructura

```
Proy1_28059781_30175126/
├── PIMES.py          # script CLI (modos -e y -p)
├── PIMES.ipynb       # cuaderno con el mismo flujo explicado paso a paso
├── README.md         # este archivo
├── modelo_PIMES.joblib   # se crea al ejecutar -e
└── curva_ROC.png         # se crea al ejecutar -e
```

## Uso

### Modo de entrenamiento

Hace *5-fold cross-validation*, reporta el AUC micro-promedio, guarda la curva ROC en `curva_ROC.png` y persiste el modelo final entrenado sobre todo el conjunto en `modelo_PIMES.joblib`.

```
python PIMES.py -e medicamentos.txt efectosSecundariosDeLosMedicamentos.txt
```

### Modo de predicción

Carga `modelo_PIMES.joblib` y, para cada medicamento del archivo de medicamentos, imprime las probabilidades de los efectos secundarios pedidos en orden descendente.

```
python PIMES.py -p medicamentosSinEfectosSecun.txt listaDeEfectosSecun.txt
```

El archivo de efectos secundarios en este modo es una lista con un efecto por línea. Los efectos deben ser un subconjunto de los usados durante el entrenamiento.

## Notas de implementación

- Los archivos tabulares tienen un desfase intencional: el encabezado contiene 881 (o 1385) nombres de columna, pero cada fila empieza con el nombre del medicamento. `pandas` detecta automáticamente que la primera columna de cada fila es el índice, lo que simplifica la carga.
- Por ser un problema multietiqueta, `MLPClassifier` aplica activación logística a cada neurona de salida y minimiza la log-loss binaria por etiqueta, exactamente el equivalente a un `sigmoid` + `binary_crossentropy` con TensorFlow/Keras.
- El AUC reportado es micro-promedio: se aplanan las matrices de etiquetas verdaderas y predicciones antes de calcular una sola curva ROC. Es la métrica habitual para resumir el desempeño multietiqueta cuando hay etiquetas raras.
