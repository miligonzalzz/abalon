from pathlib import Path
from datetime import datetime
import csv
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Guardamos las direcciones importantes.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "abalone_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model_v3.pkl"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics_v3.json"
EXPERIMENTS_PATH = PROJECT_ROOT / "results" / "experiments_log.csv"

# Mismas reglas que v1 y v2 para que las métricas sean comparables
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_VERSION = "v3"
EXPERIMENT_ID = "exp_nn_v3"


def main():
    # Cargar los datos
    df = pd.read_csv(DATA_PATH)

    # Separar variables predictoras y objetivo
    # Se excluye Rings porque age = Rings + 1.5 (evita fuga de información)
    X = df.drop(columns=["Rings", "age"])
    y = df["age"]

    # Dividir en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Red neuronal pequeña (2 capas ocultas) con escalado dentro de un Pipeline:
    # el escalador se ajusta solo con los datos de entrenamiento
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            early_stopping=True,
            random_state=RANDOM_STATE
        ))
    ])
    model.fit(X_train, y_train)

    # Calcular métricas en test
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "R2": float(r2_score(y_test, y_pred))
    }

    # Hiperparámetros: solo los de la red (los del Pipeline no son serializables a JSON)
    hyperparameters = model.named_steps["mlp"].get_params()

    # Crear carpetas necesarias
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Guardar modelo comprimido (el Pipeline completo: scaler + red neuronal)
    joblib.dump(model, MODEL_PATH, compress=3)

    # Guardamos métricas, hiperparámetros y semilla en fomato JSON
    results = {
        "version": MODEL_VERSION,
        "target": "age",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "model": "Pipeline(StandardScaler + MLPRegressor)",
        "hyperparameters": hyperparameters,
        "metrics": metrics
    }

    METRICS_PATH.write_text(
        json.dumps(results, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    # Registrar la ejecución en el CSV
    experiment = {
        "experiment_id": EXPERIMENT_ID,
        "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"]
    }

    write_header = not EXPERIMENTS_PATH.exists()

    with EXPERIMENTS_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=experiment.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(experiment)

    # Mostrar resultados
    print(f"Experimento registrado: {EXPERIMENT_ID}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2: {metrics['R2']:.4f}")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")
    print(f"Registro guardado en: {EXPERIMENTS_PATH}")


if __name__ == "__main__":
    main()
