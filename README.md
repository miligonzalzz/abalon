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