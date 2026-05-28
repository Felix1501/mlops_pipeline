import pandas as pd
import numpy as np

from pathlib import Path

from scipy.stats import ks_2samp

# =========================
# CARGA DE DATOS
# =========================

ruta = Path("data/raw/Base_de_datos.xlsx")

df = pd.read_excel(ruta)

# =========================
# VARIABLES NUMERICAS
# =========================

variables_numericas = df.select_dtypes(
    include=np.number
).columns.tolist()

# =========================
# SIMULACION NUEVOS DATOS
# =========================

datos_referencia = df.copy()

datos_nuevos = df.copy()

# simulacion drift

datos_nuevos["salario_cliente"] = (
    datos_nuevos["salario_cliente"] * 1.15
)

# =========================
# MONITOREO KS TEST
# =========================

print("\n========================")
print("MONITOREO DEL MODELO")
print("========================")

for columna in variables_numericas:

    serie_referencia = datos_referencia[columna].dropna()

    serie_nueva = datos_nuevos[columna].dropna()

    if serie_referencia.nunique() <= 1 or serie_nueva.nunique() <= 1:

        estadistico = 0
        p_value = 1

    else:

        estadistico, p_value = ks_2samp(
            serie_referencia,
            serie_nueva
        )

    drift = (
        "SI"
        if p_value < 0.05
        else "NO"
    )

    print(f"\nVariable: {columna}")
    print(f"KS Statistic: {estadistico:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print(f"Drift detectado: {drift}")

# =========================
# REPORTE FINAL
# =========================

print("\nMonitoreo finalizado.")