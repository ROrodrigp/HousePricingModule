import os
import pickle
import pandas as pd
import numpy as np

from src.preprocessing.process_data import process_raw_data


def load_model(model_path):
    """
    Carga un modelo serializado en formato pickle.

    :param model_path: Ruta del archivo pickle (.pkl) del modelo entrenado.
    :return: Modelo cargado.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ERROR: El archivo de modelo '{model_path}' no existe."
        )

    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        print(f"Modelo cargado desde: {model_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"ERROR al cargar el modelo: {e}")


def make_predictions(input_file, model_file, output_file):
    """
    Procesa un CSV de entrada, carga un modelo y genera predicciones.

    :param input_file: Ruta del CSV de entrada sin procesar.
    :param model_file: Ruta del modelo serializado en pickle.
    :param output_file: Ruta del CSV donde se guardarán las predicciones.
    """
    # Validar archivo de entrada
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"ERROR: El archivo de entrada '{input_file}' no existe."
        )

    try:
        # 1. Preprocesar los datos de entrada (test)
        processed_file = "temp_processed.csv"  # Archivo temporal
        process_raw_data(input_file, processed_file)

        # 2. Cargar datos preprocesados de test
        df_processed = pd.read_csv(processed_file)
        if df_processed.empty:
            raise ValueError(
                "ERROR: El DataFrame procesado está vacío. Revisa el preprocesamiento."
            )

        # 3. Cargar el DataFrame 'prep.csv' con el que se entrenó el modelo
        train_df = pd.read_csv("data/prep.csv")
        if train_df.empty:
            raise ValueError(
                "ERROR: El DataFrame de entrenamiento (prep.csv) está vacío o no se cargó correctamente."
            )

        # --- Alinear columnas ---
        # Quitar columnas que no sean features en train_df (por ejemplo, 'Id', 'SalePrice', 'SalePrice_Log').
        train_features = train_df.drop(
            columns=["Id", "SalePrice", "SalePrice_Log"], errors="ignore"
        )

        # Para el DataFrame de test (df_processed), también quitamos 'Id'
        X_test = df_processed.drop(
            columns=["Id"], errors="ignore"
        )

        # Alinear para que X_test tenga exactamente las mismas columnas que train_features.
        train_features, X_test = train_features.align(
            X_test, join="left", axis=1)
        X_test = X_test.fillna(0)  # Rellenar con 0 los valores NaN

        # 4. Cargar modelo entrenado
        model = load_model(model_file)

        # 5. Hacer predicciones
        predictions_log = model.predict(X_test)

        # 6. Revertir la función logarítmica (asumiendo que usaste np.log1p)
        predictions = np.expm1(predictions_log)

        # 7. Guardar predicciones en un CSV
        df_predictions = pd.DataFrame({
            "Id": df_processed.get("Id", range(len(predictions))),
            "SalePrice_Predicted": predictions
        })
        df_predictions.to_csv(output_file, index=False)

        print(f"Predicciones guardadas en: {output_file}")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except ValueError as val_error:
        print(val_error)
    except Exception as e:
        print(f"ERROR inesperado: {e}")
