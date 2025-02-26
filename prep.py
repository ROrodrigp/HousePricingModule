#!/usr/bin/env python3

import argparse
from src.preprocessing.process_data import process_raw_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procesa datos crudos aplicando transformaciones y guardando el resultado.")
    parser.add_argument("--input", type=str, required=True,
                        help="Ruta del archivo CSV de entrada (raw.csv).")
    parser.add_argument("--output", type=str, required=True,
                        help="Ruta del archivo CSV de salida (prep.csv).")

    # Ejecutar el procesamiento
    args = parser.parse_args()

    process_raw_data(args.input, args.output)
