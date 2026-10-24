
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "abalone_clean.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    df = pd.read_csv(DATA_PATH)

    # Separar variables predictoras y variable objetivo
    X = df.drop(columns=["Rings", "age"])
    y = df["Rings"]

    # Dividir en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Crear y entrenar el modelo
    model = RandomForestRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    print(f"Filas de entrenamiento: {len(X_train)}")
    print(f"Filas de test: {len(X_test)}")
    print(f"Random state de la división: {RANDOM_STATE}")
    print(f"Random state del modelo: {RANDOM_STATE}")


if __name__ == "__main__":
    main()
