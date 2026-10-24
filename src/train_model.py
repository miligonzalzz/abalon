
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "abalone_clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model_v1.pkl"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics_v1.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    # Cargar los datos procesados
    df = pd.read_csv(DATA_PATH)

    # Separar variables predictoras y objetivo
    X = df.drop(columns=["Rings", "age"])
    y = df["age"]

    # División reproducible de entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Crear y entrenar el modelo
    model = RandomForestRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Predecir sobre el conjunto de test
    y_pred = model.predict(X_test)

    # Calcular métricas
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(mean_squared_error(y_test, y_pred) ** 0.5),
        "R2": float(r2_score(y_test, y_pred))
    }

    # Crear carpetas de salida
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Guardar el modelo con compresión
    joblib.dump(model, MODEL_PATH, compress=3)

    # Registrar métricas y configuración
    results = {
        "version": "v1",
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

    print(f"Filas de entrenamiento: {len(X_train)}")
    print(f"Filas de test: {len(X_test)}")
    print("\nMétricas en test:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"Métricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()
