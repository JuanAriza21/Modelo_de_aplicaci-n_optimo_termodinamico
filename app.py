import streamlit as st
import pandas as pd
import json
import base64
import os
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import auth as firebase_auth
from google.generativeai import GenerativeModel
import google.generativeai as genai
from collections import defaultdict
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms
import random

# Global variables for Firebase
__app_id = "default-app-id"
__firebase_config = '{}'
__initial_auth_token = 'None'

# Function to get the current user ID
def get_user_id():
    try:
        if 'user_id' not in st.session_state:
            auth = firebase_auth.get_auth()
            st.session_state.user_id = auth.get_user(st.session_state.token).uid
        return st.session_state.user_id
    except Exception as e:
        st.error(f"Error al obtener el ID de usuario: {e}")
        return None

# Firebase initialization
def initialize_firebase():
    global db
    if not firebase_admin._apps:
        try:
            firebase_config = json.loads(__firebase_config)
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred)
            st.session_state.is_auth_ready = False
            auth = firebase_auth.get_auth()
            firebase_auth.on_current_user_changed(auth, on_user_changed)
            st.session_state.is_auth_ready = True
        except Exception as e:
            st.error(f"Error initializing Firebase: {e}")
    else:
        st.session_state.is_auth_ready = True
    db = firestore.client()

def on_user_changed(user):
    st.session_state.is_auth_ready = True
    if user:
        st.session_state.token = firebase_auth.create_custom_token(user.uid)
        st.session_state.user_id = user.uid
    else:
        st.session_state.token = None
        st.session_state.user_id = None

# Function to connect to Gemini API
@st.cache_data
def get_gemini_model():
    api_key = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else ""
    if not api_key:
        st.warning("No se encontró la clave de la API de Gemini. La funcionalidad de análisis de IA estará limitada.")
        return None
    genai.configure(api_key=api_key)
    return GenerativeModel('gemini-2.5-flash-preview-05-20')

# Function to get the Firebase database instance
@st.cache_resource
def get_firestore_db():
    return firestore.client()

# Function to save data to Firestore
def save_data(data, collection_name, doc_id=None):
    if st.session_state.is_auth_ready and st.session_state.user_id:
        try:
            doc_ref = db.collection('artifacts').document(__app_id).collection('users').document(st.session_state.user_id).collection(collection_name)
            if doc_id:
                doc_ref.document(doc_id).set(data)
                st.success(f"Datos guardados con éxito en Firestore en el documento: {doc_id}")
            else:
                doc_ref.add(data)
                st.success(f"Datos guardados con éxito en Firestore.")
        except Exception as e:
            st.error(f"Error al guardar datos en Firestore: {e}")

# Function to load data from Firestore
def load_data(collection_name):
    if st.session_state.is_auth_ready and st.session_state.user_id:
        try:
            docs = db.collection('artifacts').document(__app_id).collection('users').document(st.session_state.user_id).collection(collection_name).stream()
            data = defaultdict(list)
            for doc in docs:
                for key, value in doc.to_dict().items():
                    data[key].append(value)
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error al cargar datos desde Firestore: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

# AI Models (placeholder for now)
def train_ann(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=1)
    model.fit(X_train, y_train)
    return model

def optimize_with_ga(data):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 100)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(data.columns))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_efficiency(individual):
        # Placeholder for a complex efficiency function
        return sum(individual) * 0.5,

    toolbox.register("evaluate", eval_efficiency)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, stats=stats, halloffame=hof, verbose=False)
    return hof[0]

# Main Streamlit app
def main():
    st.set_page_config(layout="wide")
    st.title("Sistema de Análisis Óptimo para Sistemas de Refrigeración y Calderas")

    initialize_firebase()

    if not st.session_state.get('user_id'):
        st.info("Iniciando sesión anónimamente. Por favor, espera...")
        auth = firebase_auth.get_auth()
        try:
            if __initial_auth_token and __initial_auth_token != 'None':
                firebase_auth.sign_in_with_custom_token(auth, __initial_auth_token)
            else:
                firebase_auth.sign_in_anonymously(auth)
            st.session_state.user_id = auth.current_user.uid
        except Exception as e:
            st.error(f"Error al iniciar sesión: {e}")
            st.stop()

    st.sidebar.header("Opciones de la aplicación")
    menu = ["Cargar y Analizar Datos", "Análisis con IA"]
    choice = st.sidebar.selectbox("Selecciona una opción", menu)

    if choice == "Cargar y Analizar Datos":
        st.header("Carga y Visualización de Datos")
        uploaded_file = st.file_uploader("Sube tu archivo CSV o Excel", type=["csv", "xlsx"])

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.write("### Vista previa de los datos")
                st.dataframe(df)

                if st.button("Guardar datos en Firestore"):
                    # Convert DataFrame to a list of dicts to save
                    data_to_save = df.to_dict('records')
                    save_data({"data": json.dumps(data_to_save)}, "proyectos")

            except Exception as e:
                st.error(f"Ocurrió un error al cargar el archivo: {e}")

        st.header("Cargar datos guardados")
        if st.button("Cargar datos desde Firestore"):
            with st.spinner('Cargando datos...'):
                saved_data = load_data("proyectos")
                if not saved_data.empty:
                    st.session_state.df = pd.DataFrame(json.loads(saved_data['data'][0]))
                    st.success("Datos cargados con éxito.")
                    st.dataframe(st.session_state.df)
                else:
                    st.info("No se encontraron datos guardados.")

    elif choice == "Análisis con IA":
        st.header("Análisis Avanzado con Inteligencia Artificial")
        
        if 'df' not in st.session_state:
            st.warning("Por favor, primero carga un archivo en la sección 'Cargar y Analizar Datos'.")
            return
        
        df = st.session_state.df

        st.subheader("Análisis de datos con Gemini")
        query = st.text_area("Ingresa tu consulta sobre los datos (ej: 'dame un resumen de las variables', 'analiza la relación entre la temperatura y el consumo de energía'):")
        
        if st.button("Analizar con Gemini"):
            if query:
                model = get_gemini_model()
                if model:
                    with st.spinner('Analizando datos con Gemini...'):
                        data_for_prompt = df.to_string()
                        prompt = f"Basado en los siguientes datos de un sistema de refrigeración o caldera:\n\n{data_for_prompt}\n\nResponde a la siguiente consulta de análisis de un experto: {query}"
                        
                        try:
                            response = model.generate_content(prompt)
                            st.write("### Resultados del Análisis")
                            st.write(response.text)
                        except Exception as e:
                            st.error(f"Error al conectar con la API de Gemini: {e}")
                else:
                    st.error("La clave de la API de Gemini no está configurada correctamente.")

        st.subheader("Modelos de IA basados en la Metodología")
        ai_model_choice = st.selectbox("Selecciona un modelo de IA para aplicar", ["Seleccionar...", "Redes Neuronales Artificiales (RNA)", "Algoritmos Genéticos (AG)", "Aprendizaje por Refuerzo (RL)"])

        if ai_model_choice == "Redes Neuronales Artificiales (RNA)":
            st.info("Este modelo predice el rendimiento y la eficiencia basándose en datos históricos. Asegúrate de tener variables de entrada y salida claras.")
            
            target_variable = st.selectbox("Selecciona la variable de salida (target):", df.columns)
            feature_variables = st.multiselect("Selecciona las variables de entrada (features):", [col for col in df.columns if col != target_variable])
            
            if st.button("Entrenar Modelo RNA"):
                if feature_variables and target_variable:
                    X = df[feature_variables]
                    y = df[target_variable]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    with st.spinner("Entrenando Red Neuronal..."):
                        ann_model = train_ann(X_train, y_train)
                        predictions = ann_model.predict(X_test)
                        mse = mean_squared_error(y_test, predictions)
                        st.success(f"Entrenamiento completado. Error Cuadrático Medio (MSE): {mse:.4f}")
                        
                        st.write("### Predicciones vs. Valores Reales")
                        results_df = pd.DataFrame({'Real': y_test, 'Predicción': predictions})
                        st.dataframe(results_df)
                else:
                    st.warning("Por favor, selecciona al menos una variable de entrada y una variable de salida.")

        elif ai_model_choice == "Algoritmos Genéticos (AG)":
            st.info("Este modelo optimiza parámetros para maximizar una función objetivo (ej. eficiencia, ahorro).")
            
            if st.button("Ejecutar Algoritmo Genético"):
                with st.spinner("Optimizando con Algoritmos Genéticos..."):
                    best_params = optimize_with_ga(df)
                    st.success("Optimización completada. Mejores parámetros encontrados:")
                    st.write(best_params)
        
        elif ai_model_choice == "Aprendizaje por Refuerzo (RL)":
            st.info("Este modelo es para control adaptativo en tiempo real. Su implementación es más compleja y requiere un entorno de simulación.")
            st.warning("Esta funcionalidad no está implementada en esta versión de demostración. Se podría desarrollar un entorno de simulación en el futuro.")
            
if __name__ == "__main__":
    main()
