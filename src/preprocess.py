

from pathlib import Path
from datetime import date
import json

import pandas as pd


# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "abalone_raw.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "abalone_clean.csv"
LOG_PATH = PROJECT_ROOT / "data" / "data_log.json"


def preprocess_data():
    # Cargar los datos originales
    df = pd.read_csv(RAW_PATH)

    rows_before = len(df)

    # Crear la variable de edad estimada
    df["age"] = df["Rings"] + 1.5

    # Codificar Sex con one-hot encoding
    df = pd.get_dummies(
        df,
        columns=["Sex"],
        prefix="sex",
        dtype=int
    )

    # Detectar valores atípicos mediante un criterio explícito:
    # eliminar registros con Height > 0.4
    height_threshold = 0.4
    outliers_mask = df["Height"] > height_threshold
    outliers_removed = int(outliers_mask.sum())

    df = df.loc[~outliers_mask].copy()

    # Guardar el dataset procesado
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    # Registrar las transformaciones realizadas
    log = {
        "version": "v1",
        "created_at": date.today().isoformat(),
        "source": "data/raw/abalone_raw.csv",
        "output": "data/processed/abalone_clean.csv",
        "rows_before": rows_before,
        "rows_after": len(df),
        "transformations": [
            "added derived feature 'age = Rings + 1.5'",
            "encoded Sex with one-hot encoding (sex_F, sex_I, sex_M)",
            (
                f"removed {outliers_removed} outliers "
                f"with Height > {height_threshold}"
            ),
        ],
        "parameters": {
            "age_formula": "Rings + 1.5",
            "sex_encoding": "one-hot encoding",
            "height_threshold": height_threshold,
            "outlier_rule": "remove rows where Height > 0.4",
            "scaling": None,
        },
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        json.dumps(log, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Dataset original: {rows_before} filas")
    print(f"Atípicos eliminados: {outliers_removed}")
    print(f"Dataset procesado: {df.shape}")
    print(f"CSV guardado en: {PROCESSED_PATH}")
    print(f"Registro guardado en: {LOG_PATH}")


if __name__ == "__main__":
    preprocess_data()
