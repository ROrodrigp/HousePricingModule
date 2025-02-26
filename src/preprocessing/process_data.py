"""
Módulo para el preprocesamiento de datos.

Este módulo proporciona la función `process_raw_data()` para aplicar
transformaciones al dataset y guardarlo en un nuevo archivo.
"""
import pandas as pd
from src.tools.tools_clean_data import clean_data
from src.tools.tools_encoders import apply_one_hot_encoding, apply_ordinal_encoding
from src.tools.tools_engine_variables import apply_log_transform


def process_raw_data(input_file, output_file):
    """
    Lee un archivo CSV, aplica transformaciones y guarda el resultado en otro archivo.

    :param input_file: Ruta del archivo CSV de entrada (raw.csv).
    :param output_file: Ruta del archivo CSV de salida (prep.csv).
    """
    # Leer el archivo CSV
    df = pd.read_csv(input_file)

    # Aplicar transformaciones
    df = apply_ordinal_encoding(df)
    df = apply_one_hot_encoding(df)

    # Solo transformar si la variable esta presente
    if "SalePrice" in df.columns:
        df = apply_log_transform(df, "SalePrice")

    df = clean_data(df)

    # Guardar el resultado en un nuevo archivo CSV
    df.to_csv(output_file, index=False)

    print(f"Archivo procesado guardado en: {output_file}")
