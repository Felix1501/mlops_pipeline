import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

# =========================
# CARGA DE DATOS
# =========================

ruta = Path("data/raw/Base_de_datos.xlsx")

df = pd.read_excel(ruta)

# =========================
# VARIABLE OBJETIVO
# =========================

target = "Pago_atiempo"

X = df.drop(columns=[target])

y = df[target]

# =========================
# VARIABLES NUMERICAS
# =========================

variables_numericas = X.select_dtypes(
    include=np.number
).columns.tolist()

# =========================
# VARIABLES CATEGORICAS
# =========================

variables_categoricas = X.select_dtypes(
    include="object"
).columns.tolist()

for columna in variables_categoricas:
    X[columna] = X[columna].astype(str)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# PIPE NUMERICO
# =========================

pipe_numerico = Pipeline([
    
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    
    (
        "scaler",
        StandardScaler()
    )
])

# =========================
# PIPE CATEGORICO
# =========================

pipe_categorico = Pipeline([
    
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    
    (
        "onehot",
        OneHotEncoder(handle_unknown="ignore")
    )
])

# =========================
# TRANSFORMADOR
# =========================

preprocesador = ColumnTransformer([
    
    (
        "num",
        pipe_numerico,
        variables_numericas
    ),
    
    (
        "cat",
        pipe_categorico,
        variables_categoricas
    )
])

# =========================
# TRANSFORMACION
# =========================

X_train_transformado = preprocesador.fit_transform(
    X_train
)

X_test_transformado = preprocesador.transform(
    X_test
)

# =========================
# INFORMACION FINAL
# =========================

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)

print("\nVariables numericas:")
print(variables_numericas)

print("\nVariables categoricas:")
print(variables_categoricas)

print("\nTransformacion completada.")