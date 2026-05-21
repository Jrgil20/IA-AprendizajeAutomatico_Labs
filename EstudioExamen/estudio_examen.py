"""
================================================================================
  ESTUDIO PARA EL EXAMEN — IA: Aprendizaje Automático (UCAB)
================================================================================

  Mega-archivo de práctica. Cubre los problemas vistos en Lab2, Lab3, Lab4 y
  Proyecto1. Cada bloque está pensado para COPIAR Y PEGAR en el examen ajustando
  los parámetros indicados.

  Cómo usar:
      python estudio_examen.py

  El menú te deja ejecutar cada problema y ver qué imprime. Los bloques que
  se pueden copiar al examen están delimitados con:

      # >>> PEGAR DESDE AQUÍ ====================================================
      ...
      # <<< HASTA AQUÍ ==========================================================

  Los parámetros que normalmente hay que cambiar según el problema están
  marcados con `# >>> CAMBIAR: ...`. Lee el comentario; te dice qué valor usar
  para cada lab.

  Mapa rápido de problemas:
      1) Clasificación BINARIA      (Lab2 — clasificadoresBinarios.ipynb)
      2) Clasificación MULTICLASE   (Lab2 — extensión natural)
      3) Clasificación MULTIETIQUETA (Lab2 — clasificadoresMultietiquetas.ipynb)
      4) Regresión LINEAL           (Lab3 — Act1 ventas)
      5) Regresión POLINÓMICA       (Lab3 — Act2 grado 3, Act3 Bitcoin grado 4)
      6) MLP SHALLOW (sklearn)      (Proyecto1 — PIMES)
      7) MLP Keras (referencia)     (Lab4 — interacciones medicamento-medicamento)
      8) Validación cruzada k-fold  (utilitario común)
      9) Curva ROC + AUC            (utilitario común)

================================================================================
  ÁRBOL DE DECISIÓN: ¿qué bloque uso para este ejercicio del examen?
================================================================================

  ¿Qué quieres predecir?
  │
  ├── Un NÚMERO continuo (precio, ventas, temperatura) ─────────► REGRESIÓN
  │   │
  │   ├── Relación lineal (recta) ─────────────────────► opción 4 (LINEAL)
  │   └── Relación con curvatura (sube/baja, fechas) ──► opción 5 (POLINÓMICA)
  │
  └── Una o varias CATEGORÍAS ─────────────────────────────────► CLASIFICACIÓN
      │
      ├── Una respuesta SÍ/NO ──────────────────────────► opción 1 (BINARIA)
      │
      ├── UNA clase entre N (>2) ───────────────────────► opción 2 (MULTICLASE)
      │   (cada muestra pertenece a UNA sola clase)
      │
      └── Varias etiquetas independientes a la vez ─────► opción 3 (MULTIETIQUETA)
          (cada muestra puede tener 0, 1 o varias etiquetas activas)

  ¿Y si el enunciado dice "red neuronal" o "MLP"?
      Usa la opción 6 (MLP sklearn) y configura las salidas según el caso:
          - Binaria         : Y de forma (n,)        → MLPClassifier normal
          - Multiclase      : Y de forma (n,) con valores enteros 0..C-1
          - Multietiqueta   : Y de forma (n, k) binaria → MLPClassifier multilabel
          - Regresión       : usa MLPRegressor en vez de MLPClassifier
      Si te exigen Keras explícitamente, ver opción 7.

  ¿Te piden "validación cruzada"?  Combina con opción 8.
  ¿Te piden "curva ROC" o "AUC"?    Combina con opción 9.
================================================================================
"""

# =============================================================================
#  IMPORTS — pega lo que necesites en el examen según el problema
# =============================================================================
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Datasets de juguete (siempre disponibles, no requieren descarga):
from sklearn.datasets import load_digits, fetch_openml, make_classification, make_regression

# Particionado / validación:
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict

# Pre-procesamiento:
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Modelos lineales:
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier, Ridge, Lasso

# Otros clasificadores:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Red neuronal (shallow MLP) — sklearn:
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Estrategias para multietiqueta / multiclase:
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.multioutput import MultiOutputClassifier

# Métricas:
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score, auc,
    mean_squared_error, mean_absolute_error, r2_score,
)

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RUTA_LAB3 = Path(__file__).resolve().parents[1] / "Lab3_Regresion"


# =============================================================================
#  HELPERS COMUNES — funciones cortas que reutilizan los bloques
# =============================================================================
def reporte_clasificacion(y_real, y_pred, y_score=None, etiqueta=""):
    """Imprime precision/recall/F1 (y AUC si hay puntaje continuo)."""
    print(f"\n--- Métricas {etiqueta} ---")
    print(f"  Accuracy : {accuracy_score(y_real, y_pred):.4f}")
    # Para multietiqueta y multiclase usamos promedio macro/micro
    promedio = "binary" if y_real.ndim == 1 and len(set(np.ravel(y_real))) == 2 else "macro"
    print(f"  Precision: {precision_score(y_real, y_pred, average=promedio, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(y_real, y_pred, average=promedio, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(y_real, y_pred, average=promedio, zero_division=0):.4f}")
    if y_score is not None:
        try:
            auc_v = roc_auc_score(y_real, y_score, multi_class="ovr", average="macro")
            print(f"  AUC      : {auc_v:.4f}")
        except ValueError:
            pass


def graficar_roc(y_real, y_score, titulo="Curva ROC"):
    """Grafica una curva ROC binaria (y_real 1D, y_score probabilidad clase positiva)."""
    fpr, tpr, _ = roc_curve(y_real, y_score)
    auc_v = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_v:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="azar")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return auc_v


def pausar():
    input("\nEnter para volver al menú... ")


# =============================================================================
#  1) CLASIFICACIÓN BINARIA  (Lab2 — clasificadoresBinarios.ipynb)
# =============================================================================
#  QUÉ HACE EL BLOQUE:
#    Entrena un clasificador que responde una pregunta de SÍ/NO sobre cada
#    muestra. Convierte la etiqueta original (0..9) en una etiqueta binaria
#    (1 si es el dígito objetivo, 0 si no), divide en train/test, entrena el
#    modelo, predice, e imprime accuracy / precision / recall / F1 / AUC y la
#    matriz de confusión 2x2. Opcionalmente grafica la curva ROC.
#
#  EJERCICIO DE REFERENCIA (Lab2):
#    "Construir un clasificador binario que detecte el dígito 5 entre los
#     demás dígitos del dataset MNIST/digits" usando SGDClassifier o KNN.
#     Reportar precision, recall, F1 y AUC, y graficar la curva ROC.
#
#  CÓMO IDENTIFICARLO EN EL EXAMEN (señales típicas):
#    - El enunciado dice "clasifique X vs el resto", "detecte si una imagen
#      es Y o no", "responda sí o no a la pregunta P sobre cada muestra".
#    - Sólo hay 2 clases posibles en la salida (sí / no, 1 / 0, spam / no-spam).
#    - Te piden curva ROC y AUC; o matriz de confusión 2x2; o precision/recall.
#
#  QUÉ COPIAR AL EXAMEN:
#    Todo lo que está entre `# >>> PEGAR DESDE AQUÍ` y `# <<< HASTA AQUÍ`.
#    Ajusta DIGITO_POSITIVO (clase positiva), MODELO y DATASET. Si te dan un
#    CSV en vez de digits/MNIST, reemplaza la carga por
#       df = pd.read_csv("..."); X = df.drop(columns=["etiqueta"]).values
#       y = df["etiqueta"].values
#    y construye y_bin con la regla de "clase positiva" que pidan.
#
#  >>> CAMBIAR según lo que pida el examen:
#    - DIGITO_POSITIVO: qué dígito es la clase positiva (5, 7, etc.)
#    - DATASET       : 'digits' (rápido, 8x8) o 'mnist' (lento, 28x28)
#    - MODELO        : 'sgd', 'knn', 'logreg' o 'rf'
# =============================================================================
def problema_binaria():
    print("\n" + "="*70)
    print("  PROBLEMA 1 — Clasificación BINARIA (Lab2)")
    print("="*70)

    # >>> PEGAR DESDE AQUÍ =====================================================
    DIGITO_POSITIVO = 5                # >>> CAMBIAR: la clase positiva
    DATASET = "digits"                 # >>> CAMBIAR: 'digits' o 'mnist'
    MODELO  = "sgd"                    # >>> CAMBIAR: 'sgd' | 'knn' | 'logreg' | 'rf'

    if DATASET == "digits":
        data = load_digits()
        X, y = data.data, data.target
    else:  # 'mnist' — sólo si tienes tiempo / conexión
        data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = data.data.astype(np.float32)
        y = data.target.astype(int)

    # Etiqueta binaria: 1 si es el dígito objetivo, 0 si no.
    y_bin = (y == DIGITO_POSITIVO).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_bin, test_size=0.25, random_state=42, stratify=y_bin
    )

    if MODELO == "sgd":
        clf = SGDClassifier(loss="log_loss", random_state=42, max_iter=1000)
    elif MODELO == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif MODELO == "logreg":
        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    elif MODELO == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Modelo desconocido: {MODELO}")

    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    # Puntaje continuo para ROC/AUC (depende del modelo):
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_te)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_te)
    else:
        y_score = y_pred

    reporte_clasificacion(y_te, y_pred, y_score, etiqueta=f"({MODELO} sobre {DATASET})")
    print(f"\nMatriz de confusión:\n{confusion_matrix(y_te, y_pred)}")
    # <<< HASTA AQUÍ ===========================================================

    if input("\n¿Graficar curva ROC? (s/n): ").strip().lower().startswith("s"):
        graficar_roc(y_te, y_score, titulo=f"ROC binaria — {MODELO}/{DATASET}")
    pausar()


# =============================================================================
#  2) CLASIFICACIÓN MULTICLASE  (Lab2 — extensión, 0..9 con load_digits)
# =============================================================================
#  QUÉ HACE EL BLOQUE:
#    Entrena un clasificador que asigna UNA de varias clases (más de dos) a
#    cada muestra: por ejemplo "este dígito es un 0, un 1, ..., o un 9".
#    Sklearn ya maneja multiclase de forma nativa para casi todos los modelos,
#    pero el bloque permite forzar las estrategias clásicas:
#      - 'ovr'  (One-vs-Rest): entrena un clasificador binario por clase.
#      - 'ovo'  (One-vs-One) : entrena un clasificador por cada par de clases.
#      - 'nativa': deja que el modelo lo resuelva como sepa (softmax, etc.).
#    Imprime accuracy global, matriz de confusión N×N y un reporte por clase.
#
#  EJERCICIO DE REFERENCIA (Lab2 — extensión natural):
#    "Clasifique los dígitos del 0 al 9 usando KNN / LogisticRegression /
#     RandomForest. Muestre la matriz de confusión y discuta los errores."
#
#  CÓMO IDENTIFICARLO EN EL EXAMEN:
#    - Hay más de dos clases posibles (3, 5, 10 idiomas, K especies, etc.).
#    - Cada muestra pertenece a UNA sola clase (no varias).
#    - Te piden matriz de confusión N×N o "accuracy por clase".
#    - Pueden pedir explícitamente "use OvR" u "OvO".
#
#  QUÉ COPIAR AL EXAMEN:
#    Todo entre los marcadores. Ajusta ESTRATEGIA y MODELO. Si el dataset
#    viene de un CSV, reemplaza load_digits() por
#       df = pd.read_csv("..."); y = df["clase"].values
#       X = df.drop(columns=["clase"]).values
#
#  >>> CAMBIAR:
#    - ESTRATEGIA: 'nativa', 'ovr' u 'ovo'
#    - MODELO    : igual que el problema binario
# =============================================================================
def problema_multiclase():
    print("\n" + "="*70)
    print("  PROBLEMA 2 — Clasificación MULTICLASE (digits 0..9)")
    print("="*70)

    # >>> PEGAR DESDE AQUÍ =====================================================
    ESTRATEGIA = "nativa"              # >>> CAMBIAR: 'nativa' | 'ovr' | 'ovo'
    MODELO     = "logreg"              # >>> CAMBIAR: 'sgd' | 'knn' | 'logreg' | 'rf'

    data = load_digits()
    X, y = data.data, data.target

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    base = {
        "sgd":    SGDClassifier(loss="log_loss", random_state=42, max_iter=1000),
        "knn":    KNeighborsClassifier(n_neighbors=5),
        "logreg": LogisticRegression(max_iter=2000),  # multiclase nativo en sklearn >=1.7
        "rf":     RandomForestClassifier(n_estimators=100, random_state=42),
    }[MODELO]

    if ESTRATEGIA == "ovr":
        clf = OneVsRestClassifier(base)
    elif ESTRATEGIA == "ovo":
        clf = OneVsOneClassifier(base)
    else:
        clf = base

    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    print(f"\nAccuracy global: {accuracy_score(y_te, y_pred):.4f}")
    print(f"Matriz de confusión 10x10:\n{confusion_matrix(y_te, y_pred)}")
    print(f"\nReporte detallado:\n{classification_report(y_te, y_pred, zero_division=0)}")
    # <<< HASTA AQUÍ ===========================================================
    pausar()


# =============================================================================
#  3) CLASIFICACIÓN MULTIETIQUETA  (Lab2 — clasificadoresMultietiquetas.ipynb)
# =============================================================================
#  QUÉ HACE EL BLOQUE:
#    Entrena un clasificador donde cada muestra puede tener VARIAS etiquetas
#    activas al mismo tiempo. La salida Y es una matriz binaria de forma
#    (n_muestras, n_etiquetas). Aquí se simula con dos preguntas binarias
#    independientes sobre los dígitos: "¿es par?" y "¿es >= 7?".
#    Modelos que soportan multioutput nativo: KNN, RandomForest, MLP.
#    Los que NO (SGD, LogisticRegression) se envuelven con
#    MultiOutputClassifier(modelo_base) — entrena uno por etiqueta.
#
#  EJERCICIO DE REFERENCIA (Lab2 — clasificadoresMultietiquetas.ipynb):
#    "Para cada dígito de MNIST, prediga dos etiquetas: (a) si es par y
#     (b) si es mayor o igual a 7. Use KNeighborsClassifier y reporte F1 macro."
#
#  CÓMO IDENTIFICARLO EN EL EXAMEN:
#    - Cada muestra puede pertenecer a 0, 1 o varias etiquetas a la vez.
#    - El enunciado describe DOS o más preguntas binarias independientes
#      sobre el mismo objeto ("¿tiene gripe? ¿tiene asma? ¿tiene fiebre?").
#    - La forma de Y es (n, k) con k > 1, todas binarias.
#    - Te piden F1 macro/micro, o accuracy por etiqueta.
#    - NO confundir con multiclase: multiclase = una sola clase entre varias;
#      multietiqueta = varias etiquetas independientes.
#
#  QUÉ COPIAR AL EXAMEN:
#    Todo entre los marcadores. Define tus etiquetas en `y_par` / `y_grande`
#    o crea las que sean. Si tu dataset ya viene con Y de forma (n, k), salta
#    esa parte y úsala directamente.
#
#  >>> CAMBIAR:
#    - Las dos preguntas que definen las etiquetas (par/impar, >=N, etc.)
#    - MODELO base. Si el clasificador no soporta multioutput, envuélvelo en
#      MultiOutputClassifier.
# =============================================================================
def problema_multietiqueta():
    print("\n" + "="*70)
    print("  PROBLEMA 3 — Clasificación MULTIETIQUETA (Lab2)")
    print("="*70)

    # >>> PEGAR DESDE AQUÍ =====================================================
    MODELO = "knn"                     # >>> CAMBIAR: 'knn' | 'rf' | 'sgd' | 'logreg'

    data = load_digits()
    X, y = data.data, data.target

    # >>> CAMBIAR: las dos preguntas binarias que definen las etiquetas
    y_par     = (y % 2 == 0).astype(int)     # etiqueta 1: ¿es par?
    y_grande  = (y >= 7).astype(int)         # etiqueta 2: ¿es >= 7?
    Y = np.c_[y_par, y_grande]               # forma (n, 2)

    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.25, random_state=42)

    if MODELO == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)          # soporta multi-output nativo
    elif MODELO == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif MODELO == "sgd":
        clf = MultiOutputClassifier(SGDClassifier(loss="log_loss", random_state=42, max_iter=1000))
    elif MODELO == "logreg":
        clf = MultiOutputClassifier(LogisticRegression(max_iter=2000))
    else:
        raise ValueError(MODELO)

    clf.fit(X_tr, Y_tr)
    Y_pred = clf.predict(X_te)

    print(f"\nForma de la salida: {Y_pred.shape}  (n_muestras, n_etiquetas)")
    print(f"Accuracy por etiqueta (par):    {accuracy_score(Y_te[:, 0], Y_pred[:, 0]):.4f}")
    print(f"Accuracy por etiqueta (>= 7):   {accuracy_score(Y_te[:, 1], Y_pred[:, 1]):.4f}")

    # F1 macro promediado sobre las 2 etiquetas:
    print(f"F1 macro (multietiqueta):       {f1_score(Y_te, Y_pred, average='macro'):.4f}")
    # <<< HASTA AQUÍ ===========================================================
    pausar()


# =============================================================================
#  4) REGRESIÓN LINEAL  (Lab3 — Act1, ventas)
# =============================================================================
#  QUÉ HACE EL BLOQUE:
#    Ajusta una RECTA `y = m*x + b` a un conjunto de puntos. La salida es un
#    número real (no una clase). Imprime la pendiente (coef_), el intercepto
#    (intercept_), el R² (qué tan bien explica la varianza de los datos) y
#    realiza predicciones para valores nuevos de X.
#
#  EJERCICIO DE REFERENCIA (Lab3 Act1 — datosVentas.csv):
#    "La empresa SiempreTuAmigo tiene los datos de ventas de 8 años (2018-
#     2025). Construya un modelo de regresión lineal que prediga las ventas
#     para 2026." Resultado esperado: coef≈8.88, intercept≈-17912.85, R²≈0.98,
#     predicción 2026 ≈ 80.
#
#  CÓMO IDENTIFICARLO EN EL EXAMEN:
#    - La variable a predecir es CONTINUA (precio, temperatura, ventas).
#    - Hay una sola variable predictora X (o pocas) y la relación se ve LINEAL
#      en el gráfico de dispersión (los puntos siguen una tendencia recta).
#    - Te piden coeficiente / intercepto, R² o "predecir el valor para X=...".
#    - El enunciado dice algo como "modelo de regresión lineal".
#
#  QUÉ COPIAR AL EXAMEN:
#    Todo entre los marcadores. El bloque hace `pd.read_csv` + LinearRegression
#    + predicción. Si te dan listas/arrays en vez de CSV:
#       X = np.array([2018, 2019, 2020, ...]).reshape(-1, 1)
#       y = np.array([7, 22, 25, ...])
#
#  >>> CAMBIAR:
#    - RUTA_CSV     : el dataset que te den (o array X/y a mano)
#    - X_PREDECIR   : valor(es) a predecir
#    - Si hay varias columnas predictoras: usa todas en X (regresión múltiple).
# =============================================================================
def problema_regresion_lineal():
    print("\n" + "="*70)
    print("  PROBLEMA 4 — Regresión LINEAL (Lab3 Act1)")
    print("="*70)

    # >>> PEGAR DESDE AQUÍ =====================================================
    RUTA_CSV   = RUTA_LAB3 / "datosVentas.csv"   # >>> CAMBIAR si te dan otro archivo
    X_PREDECIR = [[2026]]                        # >>> CAMBIAR: año(s) a predecir

    df = pd.read_csv(RUTA_CSV)
    print(f"Columnas del CSV: {list(df.columns)}")
    print(df.head())

    # >>> CAMBIAR: nombres reales de las columnas del CSV
    COL_X = df.columns[0]    # usa la primera columna como X por defecto
    COL_Y = df.columns[1]    # usa la segunda columna como y por defecto

    X = df[[COL_X]].values
    y = df[COL_Y].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    print(f"Coeficiente (pendiente): {modelo.coef_[0]:.4f}")
    print(f"Intercepto              : {modelo.intercept_:.4f}")
    print(f"R² sobre entrenamiento  : {modelo.score(X, y):.4f}")

    y_pred = modelo.predict(X_PREDECIR)
    for x_val, y_val in zip(X_PREDECIR, y_pred):
        print(f"Predicción para X={x_val[0]}: y = {y_val:.2f}")
    # <<< HASTA AQUÍ ===========================================================

    if input("\n¿Graficar la recta? (s/n): ").strip().lower().startswith("s"):
        plt.figure(figsize=(7, 5))
        plt.scatter(X, y, label="datos", color="C0")
        x_grid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        plt.plot(x_grid, modelo.predict(x_grid), color="C3", label="regresión lineal")
        plt.xlabel(COL_X); plt.ylabel(COL_Y); plt.legend(); plt.grid(alpha=0.3)
        plt.title("Regresión lineal")
        plt.tight_layout(); plt.show()
    pausar()


# =============================================================================
#  5) REGRESIÓN POLINÓMICA  (Lab3 — Act2 grado 3, Act3 Bitcoin grado 4)
# =============================================================================
#  QUÉ HACE EL BLOQUE:
#    Igual que regresión lineal, pero antes de ajustar transforma X en
#    [x, x², x³, ..., x^N] usando `PolynomialFeatures(degree=N)`. Luego le pasa
#    esos polinomios a un LinearRegression normal. Imprime R², MSE y predice
#    valores nuevos. La curva resultante es un polinomio de grado N.
#
#  EJERCICIOS DE REFERENCIA (Lab3):
#    - Act2: datasetProblema2.csv (100 puntos). Ajustar polinomio grado 3
#      y predecir y para X = {0, 1.5, 3, 5}.
#    - Act3: datosBitcoin.csv (366 fechas + precios). Convertir la fecha
#      MM/DD/YYYY a "número de días desde la fecha mínima", ajustar polinomio
#      grado 4, predecir el precio para una fecha dada y calcular el
#      porcentaje de desviación: |real - pred| / real * 100.
#
#  CÓMO IDENTIFICARLO EN EL EXAMEN:
#    - Variable a predecir continua, pero el scatter plot muestra CURVATURA
#      (no es una recta) — sube, baja, hace forma de U, S, etc.
#    - El enunciado dice "regresión polinómica de grado N" o pide ajustar
#      "y = a + bx + cx² + dx³ + ...".
#    - Si el eje X es una fecha y la variable a predecir oscila en el tiempo,
#      casi siempre es polinomio (o un modelo temporal).
#
#  QUÉ COPIAR AL EXAMEN:
#    Todo entre los marcadores. Las tres líneas clave son:
#       poly = PolynomialFeatures(degree=GRADO, include_bias=False)
#       X_poly = poly.fit_transform(X)
#       modelo = LinearRegression().fit(X_poly, y)
#    Y para predecir un valor nuevo `x_nuevo`:
#       y_pred = modelo.predict(poly.transform(np.array([[x_nuevo]])))[0]
#
#  PARA BITCOIN (fechas → días):
#       df["dias"] = (pd.to_datetime(df["fecha"]) - pd.to_datetime(df["fecha"]).min()).dt.days
#       X = df[["dias"]].values
#    Y para el % de desviación: pct = abs(y_real - y_pred) / y_real * 100
#
#  >>> CAMBIAR:
#    - GRADO : 3 para Act2, 4 para Act3 (Bitcoin), lo que pidan en el examen.
#    - RUTA_CSV / generación de datos sintéticos
#    - X_PREDECIR
# =============================================================================
def problema_regresion_polinomica():
    print("\n" + "="*70)
    print("  PROBLEMA 5 — Regresión POLINÓMICA (Lab3 Act2/Act3)")
    print("="*70)

    # >>> PEGAR DESDE AQUÍ =====================================================
    GRADO      = 3                          # >>> CAMBIAR: 3 (Act2) | 4 (Act3 Bitcoin)
    X_PREDECIR = np.array([[0], [1.5], [3], [5]])   # >>> CAMBIAR según el examen

    # Si te dan un CSV, úsalo:
    #   df = pd.read_csv("datosProblema2.csv")
    #   X = df[["x"]].values; y = df["y"].values
    # Aquí generamos un conjunto sintético para probar el bloque.
    rng = np.random.default_rng(0)
    X = np.linspace(-2, 5, 100).reshape(-1, 1)
    y = (0.5 * X.ravel()**3 - 2 * X.ravel()**2 + X.ravel() + 3
         + rng.normal(0, 1.5, size=X.shape[0]))

    poly = PolynomialFeatures(degree=GRADO, include_bias=False)
    X_poly = poly.fit_transform(X)

    modelo = LinearRegression()
    modelo.fit(X_poly, y)

    y_hat = modelo.predict(X_poly)
    print(f"Grado del polinomio : {GRADO}")
    print(f"R² entrenamiento    : {r2_score(y, y_hat):.4f}")
    print(f"MSE                 : {mean_squared_error(y, y_hat):.4f}")

    for x_val in X_PREDECIR:
        x_poly = poly.transform(x_val.reshape(1, -1))
        print(f"  Predicción X={x_val[0]:>5}:  y = {modelo.predict(x_poly)[0]:.4f}")
    # <<< HASTA AQUÍ ===========================================================

    # Para el Lab3 Act3 (Bitcoin) se calcula además el % de desviación:
    #   pct = abs(y_real - y_pred) / y_real * 100
    # Y se convierte fecha "MM/DD/YYYY" en días con:
    #   df["dias"] = (pd.to_datetime(df["fecha"]) - pd.to_datetime(df["fecha"]).min()).dt.days

    if input("\n¿Graficar curva ajustada? (s/n): ").strip().lower().startswith("s"):
        x_grid = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        plt.figure(figsize=(7, 5))
        plt.scatter(X, y, label="datos", color="C0", s=15)
        plt.plot(x_grid, modelo.predict(poly.transform(x_grid)),
                 color="C3", label=f"polinomio grado {GRADO}")
        plt.legend(); plt.grid(alpha=0.3)
        plt.title(f"Regresión polinómica grado {GRADO}")
        plt.tight_layout(); plt.show()
    pausar()


# =============================================================================
#  6) MLP SHALLOW MULTIETIQUETA  (Proyecto1 — PIMES)
# =============================================================================
#  QUÉ HACE EL BLOQUE:
#    Entrena un Perceptrón Multicapa (red neuronal "poco profunda") usando
#    sklearn — equivale a un Keras con capas Dense + ReLU + salida sigmoide
#    y pérdida binary_crossentropy. Usa validación cruzada k-fold: divide los
#    datos en k partes, entrena con k-1 y evalúa con la restante, y promedia
#    el AUC. Reporta el AUC micro-promedio (aplana todas las etiquetas en un
#    solo vector binario antes de calcular la curva ROC).
#
#  EJERCICIO DE REFERENCIA (Proyecto1 PIMES):
#    "Dada la estructura química de un medicamento (881 subestructuras
#     binarias), predecir la probabilidad de cada uno de los 1385 efectos
#     secundarios posibles. Use una red neuronal poco profunda y reporte el
#     AUC con validación cruzada de 5 particiones."
#    Arquitectura del proyecto: (512, 256, 128), Adam, lr=1e-3, batch=32.
#    Resultado: AUC ≈ 0.887.
#
#  CÓMO IDENTIFICARLO EN EL EXAMEN:
#    - Se menciona "red neuronal poco profunda" o "MLP" o "shallow network".
#    - Hay MUCHAS salidas binarias independientes (>10) → multietiqueta.
#    - Te piden validación cruzada (5-fold, 10-fold) y AUC.
#    - El dataset suele ser tabular y de dimensión grande (>50 features).
#
#  QUÉ COPIAR AL EXAMEN:
#    Todo entre los marcadores. Las líneas críticas son:
#       mlp = MLPClassifier(hidden_layer_sizes=(512,256,128),
#                           activation='relu', solver='adam',
#                           learning_rate_init=1e-3, max_iter=80,
#                           random_state=42)
#       mlp.fit(X_train, Y_train)            # Y_train de forma (n, n_etiquetas)
#       proba = mlp.predict_proba(X_test)    # (n_test, n_etiquetas)
#    Y para el AUC micro-promedio:
#       fpr, tpr, _ = roc_curve(Y_test.ravel(), proba.ravel())
#       auc_micro = auc(fpr, tpr)
#
#  >>> CAMBIAR:
#    - HIDDEN_LAYERS : tupla de neuronas por capa oculta
#                      (Proyecto1 usó (512,256,128); para datasets chicos basta (64,32))
#    - MAX_ITER, LR  : iteraciones y learning rate
#    - N_SPLITS      : folds de la validación cruzada
# =============================================================================
def problema_mlp_sklearn():
    print("\n" + "="*70)
    print("  PROBLEMA 6 — MLP shallow MULTIETIQUETA (Proyecto1 PIMES)")
    print("="*70)

    # >>> PEGAR DESDE AQUÍ =====================================================
    HIDDEN_LAYERS = (64, 32)               # >>> CAMBIAR: ej. (512,256,128) en el examen
    MAX_ITER      = 100
    LR            = 1e-3
    N_SPLITS      = 5

    # Dataset sintético multietiqueta (3 etiquetas) para no tardar:
    X, y = make_classification(
        n_samples=400, n_features=20, n_informative=10,
        n_classes=2, random_state=42,
    )
    # Convertimos en multietiqueta artificial: 3 preguntas distintas sobre X
    Y = np.c_[
        (X[:, 0] > 0).astype(int),
        (X[:, 1] + X[:, 2] > 0).astype(int),
        (X[:, 3] < X[:, 4]).astype(int),
    ]

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    aucs = []
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        mlp = MLPClassifier(
            hidden_layer_sizes=HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            learning_rate_init=LR,
            max_iter=MAX_ITER,
            random_state=42,
        )
        mlp.fit(X[tr], Y[tr])
        proba = mlp.predict_proba(X[te])      # forma (n, n_labels)
        fpr, tpr, _ = roc_curve(Y[te].ravel(), proba.ravel())
        a = auc(fpr, tpr)
        aucs.append(a)
        print(f"  Fold {fold}: AUC micro = {a:.4f}")

    print(f"\nAUC promedio: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    # <<< HASTA AQUÍ ===========================================================
    pausar()


# =============================================================================
#  7) MLP CON KERAS  (Lab4 — referencia, no ejecutable en Python 3.14)
# =============================================================================
#  QUÉ HACE EL BLOQUE:
#    Versión en Keras/TensorFlow del MLP poco profundo. Cada capa Dense es una
#    matriz de pesos. La pérdida y la activación de salida cambian según el
#    tipo de problema:
#       - MULTIETIQUETA (varias etiquetas binarias):
#             salida='sigmoid', loss='binary_crossentropy'
#       - MULTICLASE (una clase de varias, one-hot):
#             salida='softmax', loss='categorical_crossentropy'
#         (o 'sparse_categorical_crossentropy' si y es entero, no one-hot)
#       - BINARIO (una salida):
#             salida='sigmoid', loss='binary_crossentropy'
#       - REGRESIÓN:
#             salida=None (linear), loss='mse'
#
#  EJERCICIO DE REFERENCIA (Lab4 — predictorInteraccionMM.ipynb):
#    "Dada la matriz de distancias químicas de 548 medicamentos, predecir
#     con qué medicamentos interactúa cada uno (matriz binaria 548×548)."
#    Arquitectura del Lab4: Dense(512)→Dense(256)→Dense(128)→Dense(548, sigmoid),
#    Adam(lr=1e-4), 200 epochs, batch_size=32, loss='binary_crossentropy'.
#
#  CÓMO IDENTIFICARLO EN EL EXAMEN:
#    - El enunciado dice "use Keras" o "use TensorFlow" explícitamente.
#    - Pide construir una arquitectura específica (capas, activaciones).
#    - Pide graficar la curva de pérdida / accuracy por epoch.
#    - Usa `model.fit(...)`, `model.predict(...)`, `model.summary()`.
#
#  QUÉ COPIAR AL EXAMEN:
#    El bloque completo entre los marcadores. Lo más importante es la receta
#    para elegir activación + loss según el tipo de problema (ver arriba).
#    NOTA: este bloque NO corre acá porque tu Python 3.14 no tiene TF; si
#    el examen sí tiene TF instalado, se copia tal cual.
#
#  >>> CAMBIAR:
#    - input_dim         : número de features
#    - número de capas y unidades
#    - activación de salida ('sigmoid' multietiqueta, 'softmax' multiclase)
#    - loss              : 'binary_crossentropy' (multietiqueta) |
#                          'categorical_crossentropy' (multiclase) |
#                          'mse' (regresión)
# =============================================================================
def problema_mlp_keras_referencia():
    print("\n" + "="*70)
    print("  PROBLEMA 7 — MLP Keras (Lab4) — BLOQUE DE REFERENCIA")
    print("="*70)
    print("""
# >>> PEGAR DESDE AQUÍ =====================================================
import tensorflow as tf
from tensorflow import keras

# >>> CAMBIAR: dimensiones según el dataset del examen
N_FEATURES = 881
N_SALIDAS  = 1385

modelo = keras.Sequential([
    keras.layers.Input(shape=(N_FEATURES,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(N_SALIDAS, activation='sigmoid'),   # 'softmax' si multiclase
])

modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',           # 'categorical_crossentropy' si multiclase
    metrics=['accuracy'],
)
modelo.summary()

historia = modelo.fit(X_train, Y_train,
                      validation_split=0.2,
                      epochs=200,
                      batch_size=32,
                      verbose=0)

probas = modelo.predict(X_test)            # (n, N_SALIDAS) con probabilidades
# <<< HASTA AQUÍ ===========================================================

Equivalente sklearn que SÍ corre acá (ver opción 6):
    mlp = MLPClassifier(hidden_layer_sizes=(512,256,128),
                        activation='relu', solver='adam',
                        learning_rate_init=1e-4,
                        max_iter=200)
    mlp.fit(X_train, Y_train)
""")
    pausar()


# =============================================================================
#  8) VALIDACIÓN CRUZADA k-fold  (utilitario común)
# =============================================================================
#  QUÉ HACE EL BLOQUE:
#    Evalúa un modelo de forma más robusta que con un solo train/test split.
#    Divide los datos en k partes (folds), entrena el modelo k veces (cada
#    vez deja un fold fuera para test) y reporta el puntaje promedio ± desvío.
#    Esto evita que el resultado dependa de UN único split afortunado.
#
#  EJERCICIO DE REFERENCIA:
#    Cualquier problema que pida "validación cruzada de 5 particiones" o
#    "evalúe el modelo con k-fold". El Proyecto1 (PIMES) usa este patrón
#    explícitamente para reportar el AUC promedio.
#
#  CÓMO IDENTIFICARLO EN EL EXAMEN:
#    - "Use validación cruzada de k particiones / k-fold cross-validation".
#    - "Reporte el desempeño promedio del modelo".
#    - Quieren ver que NO sólo entrenaste con un train/test, sino con varios.
#
#  QUÉ COPIAR AL EXAMEN:
#    La línea clave es `cross_val_score(modelo, X, y, cv=K, scoring=SCORING)`.
#    Los valores típicos de `scoring`:
#       Clasificación   : 'accuracy', 'f1', 'f1_macro', 'roc_auc', 'precision', 'recall'
#       Regresión       : 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
#    (Sklearn devuelve los errores como NEGATIVO; toma -puntajes para verlo positivo.)
#
#  >>> CAMBIAR: K, MODELO, MÉTRICA (scoring)
# =============================================================================
def utilitario_kfold():
    print("\n" + "="*70)
    print("  UTILITARIO — Validación cruzada k-fold")
    print("="*70)

    # >>> PEGAR DESDE AQUÍ =====================================================
    K        = 5                                # >>> CAMBIAR: número de folds
    SCORING  = "accuracy"                       # >>> CAMBIAR: 'f1', 'roc_auc', 'r2', 'neg_mean_squared_error'
    modelo   = LogisticRegression(max_iter=2000)

    data = load_digits()
    X, y = data.data, (data.target == 5).astype(int)

    puntajes = cross_val_score(modelo, X, y, cv=K, scoring=SCORING)
    print(f"Puntajes por fold: {[f'{s:.4f}' for s in puntajes]}")
    print(f"Promedio: {puntajes.mean():.4f}  ±  {puntajes.std():.4f}")
    # <<< HASTA AQUÍ ===========================================================
    pausar()


# =============================================================================
#  9) CURVA ROC + AUC  (utilitario común)
# =============================================================================
#  QUÉ HACE EL BLOQUE:
#    Para un clasificador BINARIO, calcula la curva ROC y el AUC. La curva ROC
#    grafica la "tasa de verdaderos positivos" (TPR = recall) contra la "tasa
#    de falsos positivos" (FPR) variando el umbral de decisión. El AUC es el
#    área bajo esa curva: 1.0 = clasificador perfecto, 0.5 = azar.
#    Requiere PUNTAJES CONTINUOS, no etiquetas duras:
#       - Si el modelo tiene predict_proba: usa proba[:, 1].
#       - Si tiene decision_function (SGD, SVM): úsala directamente.
#
#  EJERCICIO DE REFERENCIA (Lab2):
#    Para los clasificadores binarios del Lab2 ("¿es 5?", "¿es par?"), se
#    grafica la ROC y se reporta el AUC. En multietiqueta o multiclase se
#    calcula el AUC micro o macro (ver Proyecto1).
#
#  CÓMO IDENTIFICARLO EN EL EXAMEN:
#    - Dice "grafique la curva ROC", "calcule el AUC", "área bajo la curva".
#    - Es CLASIFICACIÓN (no regresión).
#    - Tienes acceso a las probabilidades del modelo (predict_proba).
#
#  QUÉ COPIAR AL EXAMEN:
#    Las dos líneas clave:
#       fpr, tpr, _ = roc_curve(y_real, y_score)   # y_score = probas o decision_function
#       auc_v = auc(fpr, tpr)
#    Para graficar: plt.plot(fpr, tpr) y plt.plot([0,1],[0,1],'--').
#    En multietiqueta usa `roc_curve(Y.ravel(), proba.ravel())` para AUC micro.
# =============================================================================
def utilitario_roc():
    print("\n" + "="*70)
    print("  UTILITARIO — Curva ROC + AUC (binaria)")
    print("="*70)

    # >>> PEGAR DESDE AQUÍ =====================================================
    data = load_digits()
    X, y = data.data, (data.target == 5).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    y_score = clf.predict_proba(X_te)[:, 1]

    fpr, tpr, _ = roc_curve(y_te, y_score)
    auc_v = auc(fpr, tpr)
    print(f"AUC = {auc_v:.4f}")
    # <<< HASTA AQUÍ ===========================================================

    if input("\n¿Mostrar gráfica? (s/n): ").strip().lower().startswith("s"):
        graficar_roc(y_te, y_score, titulo="ROC — Logistic Regression (digit '5')")
    pausar()


# =============================================================================
#  MENÚ INTERACTIVO
# =============================================================================
OPCIONES = {
    "1": ("Clasificación BINARIA            (Lab2)",         problema_binaria),
    "2": ("Clasificación MULTICLASE         (Lab2 ext.)",    problema_multiclase),
    "3": ("Clasificación MULTIETIQUETA      (Lab2)",         problema_multietiqueta),
    "4": ("Regresión LINEAL                 (Lab3 Act1)",    problema_regresion_lineal),
    "5": ("Regresión POLINÓMICA             (Lab3 Act2/3)",  problema_regresion_polinomica),
    "6": ("MLP shallow multietiqueta sklearn(Proyecto1)",    problema_mlp_sklearn),
    "7": ("MLP Keras (referencia, Lab4)",                    problema_mlp_keras_referencia),
    "8": ("Utilitario: validación k-fold",                   utilitario_kfold),
    "9": ("Utilitario: curva ROC + AUC",                     utilitario_roc),
}


def menu():
    while True:
        print("\n" + "="*70)
        print("  ESTUDIO PARA EL EXAMEN — elige un problema")
        print("="*70)
        for k, (titulo, _) in OPCIONES.items():
            print(f"  {k}) {titulo}")
        print("  q) Salir")
        eleccion = input("\nOpción: ").strip().lower()
        if eleccion in ("q", "salir", "exit"):
            print("Suerte con el examen.")
            break
        if eleccion in OPCIONES:
            try:
                OPCIONES[eleccion][1]()
            except Exception as e:
                print(f"\nERROR ejecutando opción {eleccion}: {e}")
                pausar()
        else:
            print("Opción inválida.")


if __name__ == "__main__":
    menu()
