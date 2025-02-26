"""
Módulo de limpieza de datos para preprocesamiento.

Este módulo contiene funciones para limpiar un DataFrame, incluyendo
la eliminación de valores nulos y columnas no deseadas.
"""


def clean_data(df):
    """
    Realiza la limpieza de valores nulos y elimina columnas no deseadas.

    :param df: DataFrame a limpiar.
    :return: DataFrame limpio.
    """
    df_cleaned = df.copy()

    # Reemplazar nulos en LotFrontage con 0
    df_cleaned['LotFrontage'] = df_cleaned['LotFrontage'].fillna(0)

    # Eliminar filas con valores nulos en MasVnrArea
    df_cleaned = df_cleaned.dropna(subset=['MasVnrArea'])

    # Eliminar la columna GarageYrBlt
    df_cleaned = df_cleaned.drop(columns=['GarageYrBlt'])

    # Eliminar cualquier otro nulo restante
    df_cleaned = df_cleaned.dropna()


    return df_cleaned
