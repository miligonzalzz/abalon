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