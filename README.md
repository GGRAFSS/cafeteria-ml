# ☕ Cafetería Universitaria — Proyecto Integrador de Aprendizaje Automático

Sistema web desarrollado en **Python + Streamlit** que aplica técnicas de Aprendizaje Automático sobre datos de clientes de una cafetería universitaria (7:00 AM – 11:00 AM). Los datos se almacenan en **Supabase (PostgreSQL)**.

> **🔗 App en vivo:** [https://TU-APP.streamlit.app](https://TU-APP.streamlit.app)  
> **📦 Repositorio:** [https://github.com/TU-USUARIO/cafeteria-ml](https://github.com/TU-USUARIO/cafeteria-ml)

---

## 📋 Descripción del Proyecto

Este proyecto forma parte de la asignatura **Aprendizaje Automático** de la Maestría en Inteligencia Artificial (TecNM). Cubre las 4 unidades del temario:

*   **Unidad 1 (Fundamentos):** Análisis exploratorio, clasificación de variables y definición de Tarea/Experiencia/Desempeño (Mitchell, 1997).
*   **Unidad 2 (Supervisado):** Regresión (MSE, R²) y Clasificación (Regresión Logística, KNN, SVM, Árboles; Matriz de Confusión, F1-Score, ROC).
*   **Unidad 3 (No Supervisado):** Distancias (Euclidiana, Jaccard) y Clustering (K-Means, Jerárquico, DBSCAN; validado con Silhouette).
*   **Unidad 4 (Refuerzo):** Simulador de Q-Learning (Ecuación de Bellman) donde un agente aprende a optimizar la asignación de empleados en horas pico.

---

## 🗂️ Estructura del Dataset

| Variable | Tipo | Clasificación | Uso en ML |
|---|---|---|---|
| `ID_Cliente` | Entero | Numérica Discreta | Identificador (No se usa) |
| `Hora_Llegada` | Float | Numérica Continua | Feature |
| `Tiempo_Espera_Min` | Float | Numérica Continua | Feature |
| `Tiempo_Servicio_Min` | Float | Numérica Continua | Feature |
| `Tipo_Cliente` | Texto | Categórica Nominal | Feature (Encoding) |
| `Metodo_Pago` | Texto | Categórica Nominal | Feature (Encoding) |
| `Consumo_Total_MXN` | Float | Numérica Continua | **Target (Regresión)** |
| `Satisfaccion_1_al_5` | Entero | Categórica Ordinal | **Target (Clasificación)** |
| `Es_Hora_Pico` | Texto | Categórica Binaria | Feature / Target |

---

## 📁 Estructura del Repositorio

```text
cafeteria-ml/
├── .streamlit/
│   └── secrets.toml            # Credenciales (NO se sube a GitHub)
├── app.py                      # Interfaz Streamlit (6 pestañas)
├── data_manager.py             # Conexión Supabase, validación, ingesta
├── mod_u1_fundamentos.py       # Unidad 1: Teoría y EDA
├── mod_supervisado.py          # Unidad 2: Regresión y Clasificación
├── mod_no_supervisado.py       # Unidad 3: Distancias y Clustering
├── mod_refuerzo.py             # Unidad 4: Q-Learning
├── requirements.txt
└── README.md