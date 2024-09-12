
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tarfile
import urllib.request
import os
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url: str = HOUSING_URL, housing_path: str = HOUSING_PATH) -> None:
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path: str = HOUSING_PATH) -> pd.DataFrame:
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def preprocess_data(housing: pd.DataFrame) -> tuple:
    cat_attribs = ["ocean_proximity"]
    num_attribs = ["total_rooms", "total_bedrooms", "population", "households", "housing_median_age", "median_income",
                   "latitude", "longitude"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder())
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    housing_prepared = full_pipeline.fit_transform(housing)
    return housing_prepared, full_pipeline


def train_model(housing_prepared: np.ndarray, housing_labels: pd.Series) -> RandomForestRegressor:
    forest_reg = RandomForestRegressor(max_features=8, n_estimators=30)
    forest_reg.fit(housing_prepared, housing_labels)
    return forest_reg


def create_streamlit_ui() -> None:
    st.set_page_config(
        page_title="355228. Itzel Pérez",
        page_icon="💻",
        layout="wide"
    )

    st.title("Proyecto de Data Science")
    st.header("Itzel Guadalupe Pérez Montes. 355228")
    st.subheader("Análisis y predicción de precios de viviendas en el estado de California")

    st.write("Datos principales:")
    housing = load_housing_data()
    st.dataframe(housing.head())

    housing_prepared, full_pipeline = preprocess_data(housing)
    housing_labels = housing["median_house_value"].copy()

    st.header("Predicciones")

    with st.expander("Selecciona tus datos para darte una predicción"):
        col1, col2 = st.columns(2)
        with col1:
            rooms = st.slider("Número de cuartos", min_value=1, max_value=1000, value=300)
            bedrooms = st.slider("Número de dormitorios", min_value=1, max_value=1000, value=300)
            population = st.slider("Población", min_value=1, max_value=1000, value=1000)
        with col2:
            households = st.slider("Número de familias", min_value=1, max_value=1000, value=400)
            latitude = st.slider("Latitud", min_value=30.0, max_value=45.0, value=34.0)
            longitude = st.slider("Longitud", min_value=-125.0, max_value=-114.0, value=-118.0)

        housing_median_age = st.slider("Edad promedio", min_value=1, max_value=100, value=30)
        median_income = st.number_input("Salario promedio", value=5)
        proximity = st.selectbox("Cercanía al océano", ['NEAR BAY', 'INLAND', 'NEAR OCEAN', 'ISLAND', '1HR OCEAN'])

    if st.button("Predecir"):
        filter_housing = pd.DataFrame({
            'total_rooms': [rooms],
            'total_bedrooms': [bedrooms],
            'population': [population],
            'households': [households],
            'housing_median_age': [housing_median_age],
            'median_income': [median_income],
            'ocean_proximity': [proximity],
            'latitude': [latitude],
            'longitude': [longitude]
        }, index=[0])

        st.subheader("Datos seleccionados anteriormente")
        st.dataframe(filter_housing)

        new_data_prepared = full_pipeline.transform(filter_housing)

        final_model = train_model(housing_prepared, housing_labels)
        pred = final_model.predict(new_data_prepared)

        st.subheader("Estimación del precio")
        st.success(f"Costo en dólares: ${pred[0]:,.2f}")

        st.write("Precisión de la predicción", final_model.score(housing_prepared, housing_labels))


if __name__ == "__main__":
    if not os.path.exists(os.path.join(HOUSING_PATH, "housing.csv")):
        st.write("Un momento, buscando datos")
        fetch_housing_data()

    create_streamlit_ui()

