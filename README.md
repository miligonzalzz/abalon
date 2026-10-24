# Proyecto Abalone

## Descripción
Proyecto de aprendizaje automático para el análisis de datos y el entrenamiento de modelos predictivos sobre el conjunto de datos Abalone.

## Requisitos
- Python 3.13.14
- Git

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/miligonzalzz/abalon.git
   cd abalon
   ```

2. Crear un entorno virtual:
   ```bash
   python -m venv .venv
   ```

3. Activar el entorno virtual en Windows PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Instalar las dependencias:
   ```powershell
   python -m pip install -r requirements.txt
   ```

## Estructura del proyecto

- `data/raw/`: datos originales.
- `data/processed/`: datos procesados.
- `src/`: scripts de preprocesamiento y entrenamiento.
- `models/`: modelos entrenados.
- `results/`: métricas y registros de experimentos.
- `requirements.txt`: dependencias y versiones.
- `.gitignore`: archivos excluidos del control de versiones.

## Reproducibilidad
Se utiliza un entorno virtual y un archivo `requirements.txt` con versiones fijadas para facilitar la instalación de las dependencias.
## Preprocesamiento

Para ejecutar el preprocesamiento desde la raíz del proyecto, correr:

```powershell
.\.venv\Scripts\python.exe src/preprocess.py
```

El script lee el dataset original desde `data/raw/abalone_raw.csv` y genera:

- `data/processed/abalone_clean.csv`: dataset procesado.
- `data/data_log.json`: registro de la versión, las transformaciones aplicadas y los parámetros utilizados.

Las transformaciones incluyen la creación de la variable `age = Rings + 1.5`, la codificación one-hot de `Sex` y la eliminación de registros con `Height > 0.4`. No se aplica escalado de variables en esta etapa.
## Entrenamiento y evaluación del modelo

### Requisitos previos

- Python instalado.
- Entorno virtual del proyecto configurado.
- Dependencias instaladas.

### Ejecutar el entrenamiento

Desde la raíz del repositorio, ejecutar:

```powershell
.\.venv\Scripts\python.exe src/train_model.py
```

El script realiza los siguientes pasos:

1. Carga el dataset procesado desde `data/processed/abalone_clean.csv`.
2. Define `age` como variable objetivo y excluye `Rings` y `age` de las variables predictoras para evitar fuga de información.
3. Divide los datos en entrenamiento (80 %) y prueba (20 %), utilizando `random_state=42`.
4. Entrena un modelo `RandomForestRegressor`.
5. Calcula las métricas de evaluación sobre el conjunto de prueba.
6. Guarda el modelo entrenado, las métricas y el registro del experimento.

### Métricas de evaluación

Se utilizan las siguientes métricas:

- **MAE (Mean Absolute Error):** error absoluto medio entre los valores reales y las predicciones.
- **RMSE (Root Mean Squared Error):** raíz del error cuadrático medio; penaliza más los errores grandes.
- **R² (coeficiente de determinación):** mide qué proporción de la variabilidad de la variable objetivo explica el modelo.

### Resultados y artefactos

Los resultados se almacenan en:

- `models/model_v1.pkl`: modelo entrenado.
- `results/metrics_v1.json`: configuración del modelo y métricas obtenidas.
- `results/experiments_log.csv`: registro de las ejecuciones, con identificador, fecha, versión, hiperparámetros y métricas.

### Resultados de la versión v1

| Métrica | Valor |
|---|---:|
| MAE | 1.5103 |
| RMSE | 2.1422 |
| R² | 0.5692 |

Las métricas corresponden a una ejecución con `random_state=42` y una proporción de prueba del 20 %. Los resultados pueden variar si se modifican los datos, la configuración o los hiperparámetros.