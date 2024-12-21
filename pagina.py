import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu
import requests
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Depression Diagnosis Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS para personalizar el fondo y el color del texto
page_style = """
<style>
/* Fondo general de la página */
body {
    background-color: #fdddca; /* Verde menta */
    color: black; /* Letras negras */
}

/* Contenedor principal */
[data-testid="stAppViewContainer"] {
    background-color: #fdddca; /* Fondo verde menta */
    color: black; /* Letras negras */
}

/* Estilo del texto en encabezados y párrafos */
h1, h2, h3, h4, h5, h6, p {
    color: black; /* Letras negras */
}

/* Estilo del botón "Evaluate" */
div[data-testid="stButton"] > button {
    background-color: #f5f5f5; /* Fondo blanco suave */
    color: #333333; /* Texto oscuro */
    border: 2px solid #cccccc; /* Borde gris claro */
    padding: 0.5rem 1rem; /* Espaciado interno */
    border-radius: 10px; /* Bordes redondeados */
    font-size: 1rem; /* Tamaño de la fuente */
    font-weight: bold; /* Negrita */
}

/* Cambiar el estilo al pasar el cursor */
div[data-testid="stButton"] > button:hover {
    background-color: #e6e6e6; /* Fondo gris claro al hacer hover */
    color: #000000; /* Texto negro */
    border: 2px solid #999999; /* Borde más oscuro */
}
</style>
"""

# Aplicar el CSS
st.markdown(page_style, unsafe_allow_html=True)

# Barra lateral
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Information", "Diagnostic Measures", "Evaluate Data"],
        icons=["clipboard-data", "folder"],
        menu_icon="menu-button",
    )

# Función para cargar el modelo
@st.cache_resource
def load_model():
    try:
        with open('svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Error: Model file not found. Please check the file path.")
        return None
    except Exception as e:
        st.error(f"Unexpected error while loading the model: {e}")
        return None

# Cargar el modelo
model = load_model()

# Verificar si el modelo es válido
if model is not None and not hasattr(model, "predict"):
    st.error("Error: The loaded model is not valid. Please check the model file.")

# Apartado de Información
if selected == "Information":
    st.header("La depresión")
    st.subheader("¿Que es la depresión?")
    st.write(
        """
        La depresión es un trastorno mental frecuente que se caracteriza por un estado de
        ánimo persistentemente bajo, pérdida de interés o placer en actividades diarias, y
        dificultades para llevar a cabo las tareas cotidianas. Según la Organización Mundial
        de la Salud (OMS), la depresión es una de las principales causas de discapacidad a
        nivel global, afectando a más de 280 millones de personas en el mundo. Esta condición
        puede desarrollarse debido a una combinación de factores biológicos, psicológicos y
        sociales, y no debe confundirse con los cambios emocionales temporales que forman
        parte de la vida diaria (Organización Mundial de la Salud, 2023).
        """
    )

    # Mostrar imágenes relacionadas
    # Redimensionar la imagen
    img0 = Image.open("imagenes/images.jpeg")
    img0 = img0.resize((1600, 1200))

    # Posicionar con columnas
    col1, col2, col3, col4, col5 = st.columns(5)

    with col3:
        st.image(img0)

    st.subheader("Sintomas comunes")
    st.write(
        """
        Los síntomas más comunes de la depresión incluyen sentimientos persistentes de
        tristeza, irritabilidad o vacío; pérdida de interés o placer en actividades;
        alteraciones del sueño (insomnio o hipersomnia); cambios en el apetito y el peso;
        fatiga o pérdida de energía; dificultades para concentrarse; y pensamientos
        recurrentes de muerte o suicidio. Para que se considere un diagnóstico de depresión,
        estos síntomas deben durar al menos dos semanas y afectar significativamente la vida
        diaria de la persona (Mayo Clinic, 2023).
        """
    )

    # Mostrar imágenes relacionadas
    # Redimensionar la imagen
    img1 = Image.open("imagenes/images (1).jpeg")
    img2 = Image.open("imagenes/images (2).jpg")

    img1 = img1.resize((1600, 1200))
    img2 = img2.resize((1600, 1200))

    # Posicionar con columnas
    col1, col2, col3, col4, col5 = st.columns(5)

    with col2:
        st.image(img1)

    with col4:
        st.image(img2)

    st.subheader("Opciones de tratamiento")
    st.write(
        """
        El tratamiento de la depresión puede incluir psicoterapia, medicación o una
        combinación de ambos. La terapia cognitivo-conductual (TCC) es eficaz para ayudar a
        las personas a identificar y modificar patrones de pensamiento negativos. En cuanto
        a la medicación, los antidepresivos como los inhibidores selectivos de la recaptación
        de serotonina (ISRS) son comúnmente utilizados para regular los desequilibrios
        químicos en el cerebro. En casos severos, puede ser necesario recurrir a terapias
        alternativas como la estimulación magnética transcraneal o la terapia
        electroconvulsiva. Es fundamental que el tratamiento sea supervisado por un
        profesional de la salud mental y ajustado a las necesidades individuales
        (Instituto Nacional de Salud Mental, 2023).
        """
    )

    # Mostrar imágenes relacionadas
    # Redimensionar la imagen
    img3 = Image.open("imagenes/images (3).jpeg")
    img4 = Image.open("imagenes/images (4).jpg")

    img3 = img3.resize((1600, 1200))
    img4 = img4.resize((1600, 1200))

    # Posicionar con columnas
    col1, col2, col3, col4, col5 = st.columns(5)

    with col2:
        st.image(img3)

    with col4:
        st.image(img4)

# Diagnóstico Manual
if selected == "Diagnostic Measures":
    st.header('Predict Depression Diagnosis')
    st.subheader('User Input')

    # Obtener datos del usuario
    def get_user_input():
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider('Age (years)', 18, 31, 25)
            gender = st.selectbox('Gender', ['female', 'male'])
            bmi = st.slider('BMI', 0.0, 54.55, 22.0)
            who_bmi = st.selectbox('WHO BMI Category', [
                "Class I Obesity", "Class II Obesity", "Class III Obesity",
                "Normal", "Not Available", "Overweight", "Underweight"
            ])
            phq_score = st.slider('PHQ Score', 0, 24, 12)
            depression_severity = st.selectbox('Depression Severity', [
                "Mild", "Moderately severe", "None-minimal", "Moderate", 
                "Severe", "none", "Vacío (NaN)"
            ])
            depressiveness = st.selectbox('Depressiveness', ['False', 'True', "Vacío (NaN)"])
            suicidal = st.selectbox('Suicidal Thoughts', ['False', 'True', "Vacío (NaN)"])
        
        with col2:
            depression_treatment = st.selectbox('Depression Treatment', ['False', 'True', "Vacío (NaN)"])
            gad_score = st.slider('GAD Score', 0, 21, 10)
            anxiety_severity = st.selectbox('Anxiety Severity', [
                "Moderate", "Mild", "Severe", "None-minimal", "Vacío (NaN)"
            ])
            anxiousness = st.selectbox('Anxiousness', ['False', 'True', "Vacío (NaN)"])
            anxiety_diagnosis = st.selectbox('Anxiety Diagnosis', ['False', 'True', "Vacío (NaN)"])
            anxiety_treatment = st.selectbox('Anxiety Treatment', ['False', 'True', "Vacío (NaN)"])
            epworth_score = st.slider('Epworth Score', 0, 33, 16)
            sleepiness = st.selectbox('Sleepiness', ['False', 'True', "Vacío (NaN)"])

        # Mapear valores categóricos
        user_data = {
            "age": age,
            "gender": 1 if gender == 'male' else 0,
            "bmi": bmi,
            "who_bmi": {
                "Class I Obesity": 0, "Class II Obesity": 1, "Class III Obesity": 2,
                "Normal": 3, "Not Available": 4, "Overweight": 5, "Underweight": 6
            }[who_bmi],
            "phq_score": phq_score,
            "depression_severity": {
                "Mild": 0, "Moderately severe": 2, "None-minimal": 3, "Moderate": 1,
                "Severe": 5, "none": 6, "Vacío (NaN)": 4
            }[depression_severity],
            "depressiveness": {"False": 0, "True": 1, "Vacío (NaN)": 2}[depressiveness],
            "suicidal": {"False": 0, "True": 1, "Vacío (NaN)": 2}[suicidal],
            "depression_treatment": {"False": 0, "True": 1, "Vacío (NaN)": 2}[depression_treatment],
            "gad_score": gad_score,
            "anxiety_severity": {
                "Moderate": 2, "Mild": 1, "Severe": 4, "None-minimal": 3, "Vacío (NaN)": 0
            }[anxiety_severity],
            "anxiousness": {"False": 0, "True": 1, "Vacío (NaN)": 2}[anxiousness],
            "anxiety_diagnosis": {"False": 0, "True": 1, "Vacío (NaN)": 2}[anxiety_diagnosis],
            "anxiety_treatment": {"False": 0, "True": 1, "Vacío (NaN)": 2}[anxiety_treatment],
            "epworth_score": epworth_score,
            "sleepiness": {"False": 0, "True": 1, "Vacío (NaN)": 2}[sleepiness]
        }
        return pd.DataFrame(user_data, index=[0])

    user_input = get_user_input()
    if st.button("Evaluate"):
        if model is None:
            st.error("The model could not be loaded.")
        else:
            try:
                prediction = model.predict(user_input)
                st.subheader("Result")
                st.success("Depressed" if prediction[0] == 1 else "Not Depressed")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# Evaluar dataset
if selected == "Evaluate Data":
    st.header('Batch Evaluation')
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:", df.head())

        # Validar valores faltantes en 'epworth_score' y reemplazarlos con 33
        if 'epworth_score' in df.columns:
            df['epworth_score'] = df['epworth_score'].fillna(33)

        # Predicción por lotes
        if model:
            try:
                df['depression_diagnosis'] = model.predict(df.drop(columns=["depression_diagnosis"], errors='ignore'))
                st.write("Predicted Dataset:", df)
                st.info("Nota: los resultados están en la última columna con el nombre de 'depression_diagnosis'.")
            except Exception as e:
                st.error(f"Error during batch prediction: {e}")
        else:
            st.error("Model not loaded.")
