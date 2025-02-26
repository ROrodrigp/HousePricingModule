from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


def apply_ordinal_encoding(df):
    """
    Aplica el encoding ordinal a un DataFrame utilizando un diccionario de mapeo predefinido.

    - Reemplaza NaN con "None" solo en las columnas que incluyen 'None' en su lista de categorías.
    - Elimina filas que aún tengan NaN en cualquiera de las columnas ordinales.
    - Ajusta y transforma dichas columnas con OrdinalEncoder.

    :param df: DataFrame a transformar.
    :return: DataFrame con las variables ordinales codificadas.
    """
    ordinal_mappings = {
        "ExterQual":    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "ExterCond":    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "BsmtQual":     ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "BsmtCond":     ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "BsmtExposure": ['None', 'No', 'Mn', 'Av', 'Gd'],
        "BsmtFinType1": ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        "BsmtFinType2": ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
        "HeatingQC":    ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "KitchenQual":  ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "FireplaceQu":  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "GarageQual":   ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "GarageCond":   ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        "GarageFinish": ['None', 'Unf', 'RFn', 'Fin'],
        "PoolQC":       ['None', 'Fa', 'TA', 'Gd', 'Ex'],
        "Fence":        ['None', 'MnWw', 'MnPrv', 'GdWo', 'GdPrv'],
        "LotShape":     ['IR3', 'IR2', 'IR1', 'Reg'],
        "LandSlope":    ['Gtl', 'Mod', 'Sev'],
        "Utilities":    ['NoSeWa', 'AllPub'],
        "PavedDrive":   ['N', 'P', 'Y']
    }

    df_encoded = df.copy()

    # Reemplazar valores NaN con "None" solo en las columnas donde si existe None en el mapeo
    for col in ordinal_mappings.keys():
        if 'None' in ordinal_mappings[col]:
            df_encoded[col] = df_encoded[col].fillna(
                'None')

    # Eliminar filas que todavía tengan NaN en cualquiera de las columnas ordinales
    df_encoded.dropna(subset=ordinal_mappings.keys(), inplace=True)

    # Aplicar Ordinal Encoding
    encoder = OrdinalEncoder(
        categories=[ordinal_mappings[col] for col in ordinal_mappings])
    df_encoded[list(ordinal_mappings.keys())] = encoder.fit_transform(
        df_encoded[list(ordinal_mappings.keys())])

    return df_encoded


def apply_one_hot_encoding(df):
    """
    Aplica One-Hot Encoding a las columnas nominales de un DataFrame.

    :param df: DataFrame a transformar.
    :return: DataFrame con las variables nominales codificadas con One-Hot Encoding.
    """
    nominal_columns = [
        'MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig',
        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
        'Foundation', 'Heating', 'CentralAir', 'Electrical', 'Functional',
        'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition'
    ]

    df_encoded = df.copy()

    # Reemplazar valores NaN con "None" antes de aplicar One-Hot Encoding
    for col in nominal_columns:
        df_encoded[col] = df_encoded[col].fillna('None')

    # Aplicar One-Hot Encoding
    df_encoded = pd.get_dummies(df_encoded, columns=nominal_columns)

    return df_encoded
