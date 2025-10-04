import streamlit as st
import pandas as pd
import numpy as np
import json
import firebase_admin
from firebase_admin import credentials, firestore, auth
from collections import defaultdict

# --- Configuración Inicial y Conexión a Firebase ---

# Usar secretos de Streamlit para la configuración de Firebase
# El usuario debe configurar esto en su cuenta de Streamlit Cloud
try:
    firebase_secrets = st.secrets["firebase"]
    firebase_config_dict = dict(firebase_secrets)
except (FileNotFoundError, KeyError):
    # Fallback para desarrollo local si no se encuentran los secretos
    st.warning("Secretos de Firebase no encontrados. Usando configuración local (si existe).")
    # Para desarrollo local, puedes tener un archivo 'secrets.json'
    try:
        with open('secrets.json') as f:
            firebase_config_dict = json.load(f)
    except FileNotFoundError:
        firebase_config_dict = {}
        st.error("No se encontró el archivo de configuración 'secrets.json' para desarrollo local.")


# Función para inicializar Firebase de forma segura (cacheada para eficiencia)
@st.cache_resource
def initialize_firebase():
    """Inicializa la app de Firebase Admin si no ha sido inicializada antes."""
    if not firebase_admin._apps:
        try:
            # Usar un diccionario directamente es más robusto
            cred = credentials.Certificate(firebase_config_dict)
            firebase_admin.initialize_app(cred)
            st.info("Conexión con Firebase establecida.")
        except Exception as e:
            st.error(f"Error al inicializar Firebase: {e}. Asegúrate de que tus secretos estén bien configurados.")
            return None
    return firestore.client()

# Inicializar DB
db = initialize_firebase()

# --- Funciones de Base de Datos (Firestore) ---

def get_user_id():
    """Obtiene el ID del usuario actual de forma segura."""
    # En un entorno real, aquí iría la lógica de autenticación de Streamlit.
    # Por simplicidad, usaremos un ID de usuario anónimo/fijo.
    return "anonymous_user_for_demo"

def manage_object_profiles(action, profile_name=None, data=None):
    """Gestiona los perfiles de objetos en Firestore (Crear, Leer, Eliminar)."""
    if not db:
        st.error("La base de datos no está disponible.")
        return
    
    user_id = get_user_id()
    profiles_ref = db.collection("object_profiles").document(user_id).collection("profiles")

    if action == "save":
        try:
            profiles_ref.document(profile_name).set(data)
            st.success(f"Perfil '{profile_name}' guardado con éxito.")
            return True
        except Exception as e:
            st.error(f"Error al guardar el perfil: {e}")
            return False

    elif action == "load":
        try:
            profiles = profiles_ref.stream()
            return {profile.id: profile.to_dict() for profile in profiles}
        except Exception as e:
            st.error(f"Error al cargar perfiles: {e}")
            return {}

    elif action == "delete":
        try:
            profiles_ref.document(profile_name).delete()
            st.success(f"Perfil '{profile_name}' eliminado.")
            return True
        except Exception as e:
            st.error(f"Error al eliminar el perfil: {e}")
            return False

# --- Módulo: Análisis de Temperatura por Sensor ---

def page_sensor_analysis():
    """Página dedicada al análisis de datos de temperatura de sensores."""
    st.header("🌡️ Análisis de Temperatura por Sensor")
    st.write("Define perfiles para tus objetos, carga datos de un sensor y analiza la eficiencia térmica.")

    # Sección para administrar perfiles de objetos
    with st.expander("Administrar Perfiles de Objetos", expanded=False):
        st.subheader("Crear o Actualizar Perfil")
        
        # Cargar perfiles existentes para la edición
        profiles = manage_object_profiles("load")
        profile_to_edit = st.selectbox("O selecciona un perfil existente para editar", ["Crear nuevo..."] + list(profiles.keys()))

        if profile_to_edit != "Crear nuevo...":
            profile_data = profiles[profile_to_edit]
            profile_name_default = profile_to_edit
            temp_min_default = profile_data.get("temp_min")
            temp_max_default = profile_data.get("temp_max")
        else:
            profile_name_default = ""
            temp_min_default = 0.0
            temp_max_default = 0.0

        with st.form("profile_form"):
            profile_name = st.text_input("Nombre del Objeto/Perfil", value=profile_name_default)
            temp_min = st.number_input("Temperatura Óptima Mínima (°C)", value=temp_min_default, format="%.2f")
            temp_max = st.number_input("Temperatura Óptima Máxima (°C)", value=temp_max_default, format="%.2f")
            
            submitted = st.form_submit_button("Guardar Perfil")
            if submitted:
                if not profile_name:
                    st.warning("El nombre del perfil no puede estar vacío.")
                elif temp_min >= temp_max:
                    st.warning("La temperatura mínima debe ser menor que la máxima.")
                else:
                    data = {"temp_min": temp_min, "temp_max": temp_max}
                    manage_object_profiles("save", profile_name, data)

        st.subheader("Perfiles Guardados")
        profiles = manage_object_profiles("load") # Recargar por si hubo cambios
        if not profiles:
            st.info("Aún no has guardado ningún perfil.")
        else:
            for name, data in profiles.items():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                col1.write(f"**{name}**")
                col2.metric("T. Mínima", f"{data['temp_min']} °C")
                col3.metric("T. Máxima", f"{data['temp_max']} °C")
                if col4.button("Eliminar", key=f"del_{name}"):
                    manage_object_profiles("delete", name)
                    st.rerun()

    st.divider()

    # Sección para cargar y analizar datos
    st.subheader("Analizar Datos del Sensor")
    profiles = manage_object_profiles("load")
    if not profiles:
        st.info("Primero debes crear al menos un perfil de objeto para poder realizar un análisis.")
        return

    selected_profile_name = st.selectbox("1. Selecciona el perfil del objeto a analizar", list(profiles.keys()))
    
    uploaded_file = st.file_uploader("2. Sube el archivo de datos del sensor (CSV o Excel)", type=["csv", "xlsx"])

    if uploaded_file and selected_profile_name:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("### Vista Previa de los Datos Cargados")
            st.dataframe(df.head())

            temp_column = st.selectbox("3. Selecciona la columna de temperatura", df.columns)

            if st.button("🚀 Analizar Ahora", type="primary"):
                analyze_temperature_data(df, temp_column, profiles[selected_profile_name])

        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")

def analyze_temperature_data(df, temp_column, profile):
    """Realiza y muestra el análisis de temperatura."""
    st.subheader(f"Resultados del Análisis para '{temp_column}'")
    
    temp_data = pd.to_numeric(df[temp_column], errors='coerce').dropna()
    
    if temp_data.empty:
        st.error(f"La columna '{temp_column}' no contiene datos numéricos válidos.")
        return

    temp_min_opt = profile['temp_min']
    temp_max_opt = profile['temp_max']

    # 1. Resumen Estadístico
    st.write("#### Resumen Estadístico")
    st.dataframe(temp_data.describe())

    # 2. Análisis de Rango Óptimo
    total_readings = len(temp_data)
    within_range = temp_data[(temp_data >= temp_min_opt) & (temp_data <= temp_max_opt)].count()
    below_range = temp_data[temp_data < temp_min_opt].count()
    above_range = temp_data[temp_data > temp_max_opt].count()

    p_within = (within_range / total_readings) * 100
    p_below = (below_range / total_readings) * 100
    p_above = (above_range / total_readings) * 100

    st.write("#### Cumplimiento del Rango Óptimo")
    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Dentro del Rango", f"{p_within:.2f}%", help=f"{within_range} de {total_readings} mediciones")
    col2.metric("📉 Por Debajo del Rango", f"{p_below:.2f}%", help=f"{below_range} de {total_readings} mediciones")
    col3.metric("📈 Por Encima del Rango", f"{p_above:.2f}%", help=f"{above_range} de {total_readings} mediciones")

    # 3. Tabla de Frecuencias
    st.write("#### Tabla de Frecuencias")
    try:
        # Crear bins (intervalos) para la tabla de frecuencias
        bins = np.histogram_bin_edges(temp_data, bins='auto')
        freq_table = pd.cut(temp_data, bins=bins).value_counts().sort_index().reset_index()
        freq_table.columns = ['Rango de Temperatura (°C)', 'Frecuencia (Nº de mediciones)']
        st.dataframe(freq_table)
    except Exception as e:
        st.warning(f"No se pudo generar la tabla de frecuencias: {e}")


    # 4. Histograma
    st.write("#### Distribución de Temperaturas (Histograma)")
    st.bar_chart(temp_data)


# --- Módulo: Análisis Avanzado (Código Original) ---

def page_project_analysis():
    """Página que contiene la funcionalidad original de análisis avanzado con IA."""
    st.header("🤖 Análisis Avanzado de Proyectos con IA")
    st.write("Carga un conjunto de datos completo y utiliza modelos de IA para un análisis profundo.")
    # Aquí puedes pegar el resto de tu código original (Gemini, Redes Neuronales, etc.)
    # Por ahora, se deja un placeholder.
    st.info("Esta sección conservará las capacidades originales de análisis con Gemini, Redes Neuronales y Algoritmos Genéticos.")
    
    # Placeholder para la lógica de carga de archivos del modo original
    uploaded_file = st.file_uploader("Sube tu archivo de proyecto (CSV o Excel)", type=["csv", "xlsx"], key="advanced_upload")
    if uploaded_file:
        st.success(f"Archivo '{uploaded_file.name}' cargado. Funcionalidad de IA estaría disponible aquí.")
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.dataframe(df.head())
        st.write("---")
        st.subheader("Opciones de IA (Funcionalidad Original)")
        st.selectbox("Selecciona un modelo de IA", ["Análisis con Gemini", "Redes Neuronales (RNA)", "Algoritmos Genéticos (AG)"])
        st.button("Ejecutar Análisis Avanzado")


# --- Aplicación Principal ---

def main():
    """Función principal que renderiza la aplicación Streamlit."""
    st.set_page_config(
        page_title="Análisis Termodinámico",
        page_icon="🔥",
        layout="wide"
    )

    st.title("Sistema de Análisis Termodinámico Optimizado")

    if not db:
        st.error("La aplicación no puede funcionar sin una conexión a la base de datos. Revisa la configuración.")
        st.stop()
        
    st.sidebar.title("Módulos de Análisis")
    
    menu = {
        "Análisis de Temperatura por Sensor": page_sensor_analysis,
        "Análisis Avanzado de Proyectos": page_project_analysis,
    }
    
    choice = st.sidebar.radio("Selecciona un módulo", list(menu.keys()))

    # Ejecutar la página seleccionada
    menu[choice]()

if __name__ == "__main__":
    main()
