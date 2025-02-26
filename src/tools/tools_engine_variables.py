"""
Módulo para aplicar transformación logarítmica a una variable objetivo.

Este módulo proporciona una función para transformar una columna específica
usando np.log1p, lo cual es útil en modelos de regresión para estabilizar
la varianza y mejorar la normalidad de los datos.
"""

import numpy as np


def apply_log_transform(df, target_col):
    """
    Aplica la transformación logarítmica `log1p` a la columna objetivo.

    :param df: DataFrame de entrada con los datos.
    :param target_col: Nombre de la columna a transformar.
    :return: DataFrame con una nueva columna `{target_col}_Log` que contiene la transformación.
    """
    if target_col not in df.columns:
        raise ValueError(f"La columna '{target_col}' no está en el DataFrame.")

    df_transformed = df.copy()
    df_transformed[f"{target_col}_Log"] = np.log1p(df_transformed[target_col])

    return df_transformed
