import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# CARGA DEL MODELO
# =========================

modelo = joblib.load(
    "models/model_pipeline.joblib"
)

# =========================
# FASTAPI
# =========================

app = FastAPI()

# =========================
# INPUT DE DATOS
# =========================

class ClienteInput(BaseModel):
    
    tipo_credito: int
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str

# =========================
# ENDPOINT
# =========================

@app.get("/")
def inicio():
    
    return {
        "mensaje":
        "API de prediccion funcionando correctamente"
    }

# =========================
# PREDICCION
# =========================

@app.post("/predict")
def predecir(datos: ClienteInput):
    
    datos_df = pd.DataFrame([datos.dict()])
    
    prediccion = modelo.predict(datos_df)
    
    return {
        "prediccion":
        int(prediccion[0])
    }