# Proyecto MLOps - Predicción de Pago a Tiempo

## Descripción del proyecto

Este proyecto implementa un pipeline MLOps orientado a la predicción de pago a tiempo de clientes financieros utilizando técnicas de Machine Learning.

El proyecto incluye:

- análisis exploratorio de datos (EDA),
- feature engineering,
- entrenamiento de modelos,
- monitoreo de drift,
- despliegue mediante FastAPI,
- dashboard interactivo con Streamlit,
- y contenedorización con Docker.

---

# Estructura del proyecto

```bash
mlops_pipeline/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── cargar_datos.ipynb
│   └── comprension_eda.ipynb
│
├── reports/
│
├── src/
│   ├── ft_engineering.py
│   ├── model_deploy.py
│   ├── model_monitoring.py
│   └── app.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```
#Tecnologías utilizadas
Python
Pandas
NumPy
Scikit-learn
XGBoost
FastAPI
Streamlit
Docker
Git y GitHub
Modelos implementados
Logistic Regression
Random Forest
XGBoost
Funcionalidades principales
EDA

#Se realizó análisis exploratorio para identificar:

distribuciones,
correlaciones,
valores atípicos,
y relaciones entre variables.
Feature Engineering

#Se implementó:

imputación de valores nulos,
escalado de variables,
codificación OneHotEncoder,
pipeline de preprocesamiento.
Deployment API

#La API fue desarrollada con FastAPI y permite realizar predicciones mediante endpoints REST.

#Ejecutar API
uvicorn src.model_deploy:app --reload

#Documentación Swagger:

http://127.0.0.1:8000/docs
Dashboard Streamlit

Dashboard interactivo para visualización y predicción.

Ejecutar Streamlit
streamlit run src/app.py
Monitoreo del modelo

#Se implementó monitoreo básico utilizando KS Test para detección de drift de datos.

Docker
Construcción del contenedor
docker build -t mlops_pipeline .
Ejecución
docker run -p 8501:8501 mlops_pipeline
Resultados

Los modelos obtuvieron métricas elevadas de precisión debido a la fuerte relación existente entre variables financieras y la variable objetivo.

Autor

Augusto Fernandez