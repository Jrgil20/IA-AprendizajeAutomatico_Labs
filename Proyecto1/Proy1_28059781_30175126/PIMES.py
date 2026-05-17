"""PIMES — Predictor de Interacciones de Medicamentos y Efectos Secundarios.

Uso:
    python PIMES.py -e <archivo-medicamentos> <archivo-efectosSecundarios>
    python PIMES.py -p <archivo-medicamentos> <archivo-efectosSecundarios>

En modo de entrenamiento (-e) entrena una red neuronal poco profunda (MLP) con
5-fold cross-validation, muestra la curva ROC (micro-promedio), reporta el AUC
y guarda el modelo entrenado sobre la totalidad de los datos.

En modo de predicción (-p) carga el modelo previamente entrenado y, para cada
medicamento del archivo de medicamentos, lista las probabilidades de cada
efecto secundario indicado en el archivo de efectos, ordenadas de mayor a menor.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

MODEL_PATH = Path(__file__).parent / "modelo_PIMES.joblib"
ROC_PATH = Path(__file__).parent / "curva_ROC.png"
HIDDEN_LAYERS = (512, 256, 128)
MAX_ITER = 80
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
RANDOM_STATE = 42
N_SPLITS = 5


def _leer_tabla_binaria(ruta: str) -> tuple[list[str], list[str], np.ndarray]:
    """Lee una tabla cuyo encabezado tiene N nombres de columna y cada fila tiene
    `nombre_fila + N valores binarios`.

    Pandas detecta automáticamente la columna del nombre como índice cuando
    `len(header) = len(fila_de_datos) - 1`. Devuelve los nombres de las filas,
    los nombres de las columnas y la matriz binaria (int8).
    """
    df = pd.read_csv(ruta, sep="\t", header=0, dtype=str)
    nombres_columnas = list(df.columns)
    nombres_filas = df.index.astype(str).tolist()
    matriz = df.to_numpy(dtype=np.int8)
    return nombres_filas, nombres_columnas, matriz


def leer_matriz_medicamentos(ruta: str) -> tuple[list[str], list[str], np.ndarray]:
    """Devuelve (nombres_medicamentos, subestructuras, matriz)."""
    return _leer_tabla_binaria(ruta)


def leer_matriz_efectos(ruta: str) -> tuple[list[str], list[str], np.ndarray]:
    """Devuelve (nombres_medicamentos, efectos, matriz)."""
    return _leer_tabla_binaria(ruta)


def leer_lista_efectos(ruta: str) -> list[str]:
    """Lee un archivo con un efecto secundario por línea."""
    with open(ruta, encoding="utf-8") as f:
        return [linea.strip() for linea in f if linea.strip()]


def construir_modelo() -> MLPClassifier:
    """Construye una red neuronal poco profunda multietiqueta."""
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        learning_rate_init=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        early_stopping=False,
        verbose=False,
    )


def predecir_probabilidades(modelo: MLPClassifier, X: np.ndarray) -> np.ndarray:
    """Devuelve una matriz (n_muestras, n_efectos) con probabilidades de cada efecto.

    `MLPClassifier` con `y` 2D binario hace clasificación multietiqueta con
    sigmoide en la salida. `predict_proba` devuelve probabilidades en formato
    multietiqueta como matriz directa.
    """
    proba = modelo.predict_proba(X)
    # En multilabel, predict_proba ya devuelve un ndarray (n, n_labels).
    if isinstance(proba, list):  # por compatibilidad
        proba = np.array([p[:, 1] for p in proba]).T
    return proba


def graficar_roc(y_true: np.ndarray, y_score: np.ndarray, auc_micro: float) -> None:
    """Genera y muestra la curva ROC micro-promediada."""
    fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"micro-promedio (AUC = {auc_micro:.4f})", color="C0")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="azar")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curva ROC — PIMES (5-fold CV)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROC_PATH, dpi=120)
    print(f"[entrenamiento] Curva ROC guardada en {ROC_PATH}")
    if plt.get_backend().lower() not in {"agg", "pdf", "ps", "svg"}:
        plt.show()


def modo_entrenamiento(ruta_medicamentos: str, ruta_efectos: str) -> None:
    print(f"[entrenamiento] Leyendo {ruta_medicamentos}")
    nombres_med, subestructuras, X = leer_matriz_medicamentos(ruta_medicamentos)
    print(f"[entrenamiento] Leyendo {ruta_efectos}")
    nombres_ef, efectos, Y = leer_matriz_efectos(ruta_efectos)

    if nombres_med != nombres_ef:
        # Si el orden no coincide, reordenamos Y para que cada fila de X
        # corresponda con la misma fila de Y.
        idx = {n: i for i, n in enumerate(nombres_ef)}
        try:
            orden = [idx[n] for n in nombres_med]
        except KeyError as e:
            print(
                f"Error: el medicamento {e} aparece en {ruta_medicamentos} "
                f"pero no en {ruta_efectos}.",
                file=sys.stderr,
            )
            sys.exit(1)
        Y = Y[orden]

    X = X.astype(np.float32)
    Y = Y.astype(np.int8)
    print(f"[entrenamiento] Medicamentos: {X.shape[0]}")
    print(f"[entrenamiento] Subestructuras (entradas): {X.shape[1]}")
    print(f"[entrenamiento] Efectos secundarios (salidas): {Y.shape[1]}")
    print(
        f"[entrenamiento] Densidad de etiquetas positivas: {Y.mean():.4f}"
    )

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    aucs_micro: list[float] = []
    y_true_all: list[np.ndarray] = []
    y_score_all: list[np.ndarray] = []

    inicio_global = time.time()
    for fold, (idx_train, idx_test) in enumerate(kf.split(X), start=1):
        t0 = time.time()
        modelo = construir_modelo()
        modelo.fit(X[idx_train], Y[idx_train])
        y_score = predecir_probabilidades(modelo, X[idx_test])
        y_true = Y[idx_test]

        fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
        auc_fold = auc(fpr, tpr)
        aucs_micro.append(auc_fold)
        y_true_all.append(y_true)
        y_score_all.append(y_score)

        print(
            f"[entrenamiento] Fold {fold}/{N_SPLITS}: "
            f"AUC micro = {auc_fold:.4f}  ({time.time() - t0:.1f}s)"
        )

    y_true_cat = np.concatenate(y_true_all)
    y_score_cat = np.concatenate(y_score_all)
    fpr_g, tpr_g, _ = roc_curve(y_true_cat.ravel(), y_score_cat.ravel())
    auc_global = auc(fpr_g, tpr_g)
    auc_promedio = float(np.mean(aucs_micro))
    auc_desv = float(np.std(aucs_micro))

    print()
    print(f"AUC promedio (5-fold): {auc_promedio:.4f} ± {auc_desv:.4f}")
    print(f"AUC global (concatenando predicciones de los 5 folds): {auc_global:.4f}")
    print(f"Tiempo total de validación cruzada: {time.time() - inicio_global:.1f}s")

    graficar_roc(y_true_cat, y_score_cat, auc_global)

    # Entrenamiento final sobre todo el conjunto y guardado del modelo.
    print("[entrenamiento] Entrenando modelo final sobre todos los datos...")
    t0 = time.time()
    modelo_final = construir_modelo()
    modelo_final.fit(X, Y)
    print(f"[entrenamiento] Modelo final entrenado en {time.time() - t0:.1f}s")

    bundle = {
        "modelo": modelo_final,
        "subestructuras": subestructuras,
        "efectos": efectos,
        "auc_promedio": auc_promedio,
        "auc_global": auc_global,
        "hidden_layers": HIDDEN_LAYERS,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"[entrenamiento] Modelo guardado en {MODEL_PATH}")


def modo_prediccion(ruta_medicamentos: str, ruta_efectos: str) -> None:
    if not MODEL_PATH.exists():
        print(
            f"Error: no se encontró el modelo entrenado en {MODEL_PATH}. "
            f"Ejecute primero el modo -e.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[predicción] Cargando modelo desde {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    modelo: MLPClassifier = bundle["modelo"]
    subestructuras_entrenamiento: list[str] = bundle["subestructuras"]
    efectos_entrenamiento: list[str] = bundle["efectos"]

    print(f"[predicción] Leyendo {ruta_medicamentos}")
    nombres_med, subestructuras_archivo, X = leer_matriz_medicamentos(ruta_medicamentos)

    if subestructuras_archivo != subestructuras_entrenamiento:
        print(
            "Error: las subestructuras químicas del archivo de medicamentos no "
            "coinciden con las usadas en el entrenamiento.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[predicción] Leyendo lista de efectos {ruta_efectos}")
    efectos_consulta = leer_lista_efectos(ruta_efectos)

    # Validar que sea subconjunto de los efectos vistos en entrenamiento.
    set_entrenamiento = set(efectos_entrenamiento)
    desconocidos = [e for e in efectos_consulta if e not in set_entrenamiento]
    if desconocidos:
        print(
            f"Error: los siguientes efectos no fueron vistos en el entrenamiento: "
            f"{', '.join(desconocidos[:5])}"
            + (" ..." if len(desconocidos) > 5 else ""),
            file=sys.stderr,
        )
        sys.exit(1)

    indice_efecto = {e: i for i, e in enumerate(efectos_entrenamiento)}
    columnas = [indice_efecto[e] for e in efectos_consulta]

    X = X.astype(np.float32)
    print(f"[predicción] Calculando probabilidades para {len(nombres_med)} medicamentos...")
    proba = predecir_probabilidades(modelo, X)
    proba_consulta = proba[:, columnas]

    for nombre, fila in zip(nombres_med, proba_consulta):
        print()
        print(f"=== {nombre} ===")
        orden = np.argsort(-fila)
        for j in orden:
            print(f"  {efectos_consulta[j]}\t{fila[j]:.4f}")


def main(argv: list[str]) -> None:
    if len(argv) != 4 or argv[1] not in ("-e", "-p"):
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    modo = argv[1]
    archivo_medicamentos = argv[2]
    archivo_efectos = argv[3]

    if not os.path.isfile(archivo_medicamentos):
        print(f"Error: no existe {archivo_medicamentos}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(archivo_efectos):
        print(f"Error: no existe {archivo_efectos}", file=sys.stderr)
        sys.exit(1)

    if modo == "-e":
        modo_entrenamiento(archivo_medicamentos, archivo_efectos)
    else:
        modo_prediccion(archivo_medicamentos, archivo_efectos)


if __name__ == "__main__":
    main(sys.argv)
