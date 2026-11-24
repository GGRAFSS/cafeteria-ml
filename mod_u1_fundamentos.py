"""
Módulo U1: Introducción al Aprendizaje Automático
"""
import streamlit as st
import pandas as pd

def mostrar_definicion_mitchell():
    st.markdown("### 🧠 El Problema según Tom M. Mitchell (1997)")
    st.markdown(
        """
        Un programa de computadora **aprende** de la experiencia **E** con respecto a 
        una clase de tareas **T** y una medida de desempeño **P**, si su desempeño en **T**, 
        medido por **P**, mejora con **E**.
        """
    )
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Tarea (T)**\n\nPredecir el consumo o la satisfacción de los clientes de la cafetería.")
    with c2:
        st.info("**Experiencia (E)**\n\nLos 200+ registros históricos capturados en Supabase (7:00 - 11:00 AM).")
    with c3:
        st.success("**Desempeño (P)**\n\nMétricas como Error Cuadrático Medio (MSE) para regresión, o F1-Score para clasificación.")

def mostrar_clasificacion_variables(df: pd.DataFrame):
    st.markdown("### 📊 Clasificación Estadística de nuestras Variables")
    st.markdown("Basado en el Tema 3.1, clasificamos los datos de nuestra base de datos:")
    
    data = {
        "Variable": [
            "ID_Cliente", "Hora_Llegada", "Tiempo_Espera_Min", 
            "Tiempo_Servicio_Min", "Tipo_Cliente", "Metodo_Pago", 
            "Consumo_Total_MXN", "Satisfaccion_1_al_5", "Es_Hora_Pico"
        ],
        "Tipo de Dato": [
            "Numérica (Discreta)", "Numérica (Continua)", "Numérica (Continua)",
            "Numérica (Continua)", "Categórica (Nominal)", "Categórica (Nominal)",
            "Numérica (Continua)", "Categórica (Ordinal)", "Categórica (Binaria)"
        ],
        "Uso en ML": [
            "No usar (Identificador)", "Feature", "Feature",
            "Feature", "Feature (Requiere Encoding)", "Feature (Requiere Encoding)",
            "Target (Regresión) / Feature", "Target (Clasificación Ordinal)", "Target (Clasificación Binaria)"
        ]
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

def mostrar_linea_tiempo():
    st.markdown("### 🕰️ Breve Historia del Aprendizaje Automático")
    st.markdown(
        """
        *   **1959:** Arthur Samuel acuña el término *Machine Learning* (juego de damas).
        *   **1957:** Frank Rosenblatt crea el *Perceptrón*.
        *   **1969:** Minsky y Papert publican limitaciones (Invierno de la IA).
        *   **1986:** Backpropagation (Rumelhart, Hinton, Williams).
        *   **2012:** AlexNet revoluciona la visión por computadora (Deep Learning).
        *   **2016:** AlphaGo vence al campeón mundial de Go.
        *   **Actualidad:** Modelos generativos y LLMs (GPT, BERT, Stable Diffusion).
        """
    )