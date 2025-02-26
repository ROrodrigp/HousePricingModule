#!/usr/bin/env python3

"""
Script para hacer predicciones usando un modelo entrenado.

Uso:
    python inference.py --input data/test.csv --model models/best_random_forest_model.pkl --output data/predictions.csv
"""

import argparse
from src.predictions.predictions_utils import make_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Realiza predicciones usando un modelo entrenado.")
    parser.add_argument("--input", type=str, required=True,
                        help="Ruta del archivo CSV de entrada sin procesar.")
    parser.add_argument("--model", type=str, required=True,
                        help="Ruta del modelo entrenado en formato .pkl.")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Ruta del archivo CSV de salida con predicciones.")

    args = parser.parse_args()

    try:
        make_predictions(args.input, args.model, args.output)
    except Exception as e:
        print(f"ERROR: {e}")
