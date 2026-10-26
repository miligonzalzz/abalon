
from pathlib import Path
from datetime import datetime
import csv
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "abalone_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model_v2.pkl"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics_v2.json"
EXPERIMENTS_PATH = PROJECT_ROOT / "results" / "experiments_log.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_VERSION = "v2"


def main():
    #Cargar los datos
    df = pd.read_csv(DATA_PATH)

    #Separar las variables predictoras y objetivo
    X = df.drop(columns=["Rings", "age"])
    y = df["age"]

    #Dividir en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    #Entrenar el modelo
    model = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=5 , random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Calcular métricas en test
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "R2": float(r2_score(y_test, y_pred))
    }

    # Crear carpetas necesarias
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Guardar modelo comprimido
    joblib.dump(model, MODEL_PATH, compress=3)

    # Guardar métricas, hiperparámetros y semilla en JSON
    results = {
        "version": MODEL_VERSION,
        "target": "age",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "model": "RandomForestRegressor",
        "hyperparameters": model.get_params(),
        "metrics": metrics
    }

    METRICS_PATH.write_text(
        json.dumps(results, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    # Generar un identificador para el experimento
    if EXPERIMENTS_PATH.exists():
        with EXPERIMENTS_PATH.open("r", newline="", encoding="utf-8") as file:
            previous_experiments = list(csv.DictReader(file))
        experiment_number = len(previous_experiments) + 1
    else:
        experiment_number = 1

    experiment_id = "exp_rf_v2"

    # Registrar la ejecución en el CSV
    experiment = {
        "experiment_id": experiment_id,
        "datetime": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "hyperparameters": json.dumps(model.get_params(), sort_keys=True),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"]
    }

    with EXPERIMENTS_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=experiment.keys())

        if experiment_number == 1:
            writer.writeheader()

        writer.writerow(experiment)

    # Mostrar resultados
    print(f"Experimento registrado: {experiment_id}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R2: {metrics['R2']:.4f}")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")
    print(f"Registro guardado en: {EXPERIMENTS_PATH}")


if __name__ == "__main__":
    main()
