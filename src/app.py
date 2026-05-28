import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="MLOps - Predicción de Pago",
    layout="wide"
)

st.title("Dashboard MLOps - Predicción de Pago a Tiempo")

ruta_datos = Path("data/raw/Base_de_datos.xlsx")
ruta_modelo = Path("models/model_pipeline.joblib")

df = pd.read_excel(ruta_datos)
modelo = joblib.load(ruta_modelo)

st.header("Vista general del dataset")

col1, col2, col3 = st.columns(3)

col1.metric("Cantidad de registros", df.shape[0])
col2.metric("Cantidad de columnas", df.shape[1])
col3.metric("Pagos a tiempo", int(df["Pago_atiempo"].sum()))

st.subheader("Primeros registros")
st.dataframe(df.head())

st.subheader("Distribución de la variable objetivo")
st.bar_chart(df["Pago_atiempo"].value_counts())

st.subheader("Promedio de salario por tipo laboral")
st.bar_chart(df.groupby("tipo_laboral")["salario_cliente"].mean())

st.header("Predicción individual")

tipo_credito = st.number_input("Tipo de crédito", value=4)
capital_prestado = st.number_input("Capital prestado", value=800000.0)
plazo_meses = st.number_input("Plazo en meses", value=6)
edad_cliente = st.number_input("Edad del cliente", value=35)
tipo_laboral = st.selectbox("Tipo laboral", df["tipo_laboral"].astype(str).unique())
salario_cliente = st.number_input("Salario cliente", value=3500000.0)
total_otros_prestamos = st.number_input("Total otros préstamos", value=500000.0)
cuota_pactada = st.number_input("Cuota pactada", value=150000.0)
puntaje = st.number_input("Puntaje", value=85.0)
puntaje_datacredito = st.number_input("Puntaje Datacrédito", value=800.0)
cant_creditosvigentes = st.number_input("Cantidad créditos vigentes", value=2)
huella_consulta = st.number_input("Huella consulta", value=1)
saldo_mora = st.number_input("Saldo mora", value=0.0)
saldo_total = st.number_input("Saldo total", value=1000000.0)
saldo_principal = st.number_input("Saldo principal", value=800000.0)
saldo_mora_codeudor = st.number_input("Saldo mora codeudor", value=0.0)
creditos_sectorFinanciero = st.number_input("Créditos sector financiero", value=2)
creditos_sectorCooperativo = st.number_input("Créditos sector cooperativo", value=0)
creditos_sectorReal = st.number_input("Créditos sector real", value=1)
promedio_ingresos_datacredito = st.number_input("Promedio ingresos Datacrédito", value=3400000.0)
tendencia_ingresos = st.selectbox("Tendencia ingresos", df["tendencia_ingresos"].astype(str).unique())

if st.button("Predecir"):

    entrada = pd.DataFrame([{
        "tipo_credito": tipo_credito,
        "capital_prestado": capital_prestado,
        "plazo_meses": plazo_meses,
        "edad_cliente": edad_cliente,
        "tipo_laboral": tipo_laboral,
        "salario_cliente": salario_cliente,
        "total_otros_prestamos": total_otros_prestamos,
        "cuota_pactada": cuota_pactada,
        "puntaje": puntaje,
        "puntaje_datacredito": puntaje_datacredito,
        "cant_creditosvigentes": cant_creditosvigentes,
        "huella_consulta": huella_consulta,
        "saldo_mora": saldo_mora,
        "saldo_total": saldo_total,
        "saldo_principal": saldo_principal,
        "saldo_mora_codeudor": saldo_mora_codeudor,
        "creditos_sectorFinanciero": creditos_sectorFinanciero,
        "creditos_sectorCooperativo": creditos_sectorCooperativo,
        "creditos_sectorReal": creditos_sectorReal,
        "promedio_ingresos_datacredito": promedio_ingresos_datacredito,
        "tendencia_ingresos": tendencia_ingresos
    }])

    prediccion = modelo.predict(entrada)[0]

    if prediccion == 1:
        st.success("Predicción: el cliente pagaría a tiempo.")
    else:
        st.error("Predicción: el cliente podría no pagar a tiempo.")