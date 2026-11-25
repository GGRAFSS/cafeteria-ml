"""
App Streamlit: Cafetería Universitaria — Aprendizaje Automático
Proyecto Integrador - Maestría en Inteligencia Artificial

v4.0:
- Sidebar con 4 KPIs dinámicos del dataset (consumo, satisfacción,
  % hora pico y outliers) calculados en memoria, sin generar el PDF.
- Nombres institucionales corregidos + docente en el sidebar.

v3.2:
- Sidebar con dos reportes (completo + resumen 1 página).
- Estado visible del caché IA + botón para limpiarlo.
- Recargar datos limpia caché IA también.
"""
import streamlit as st
import pandas as pd
import data_manager as dm
import mod_u1_fundamentos as u1
import mod_supervisado as ms
import mod_no_supervisado as mns
import mod_refuerzo as mr
from datetime import datetime

# ============================================================
# DEBUG — Verificar que Streamlit Cloud lee el código nuevo
# ============================================================
st.write(
    f"🔍 Versión data_manager: "
    f"{'CON FIX' if hasattr(dm, '_clean_value') and 'forzando IDs nuevos' in dm.bulk_insert.__doc__ else 'VIEJA'}"
)

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Cafetería Universitaria — ML",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CARGA DE DATOS
# ============================================================
df, load_error = dm.load_data()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/coffee-to-go.png", width=80)
    st.markdown("### 📋 Proyecto Integrador")
    st.markdown(
        """
        **Materia:** Aprendizaje Automático  
        **Posgrado:** Maestría en Inteligencia Artificial  
        **Dataset:** Cafetería Universitaria
        """
    )
    st.divider()
    st.markdown("### 👤 Autores")
    st.markdown(
        """
        Rodríguez Avilez Selene  
        Raygoza Flores Gabriel  
        
        Instituto Tecnológico de Tijuana - TecNM  
        Periodo: 2026

        **Docente:** Dra. Claudia Ibeth González Berrelleza
        """
    )
    st.divider()
    st.markdown("### 🔌 Estado de la Base de Datos")
    if load_error:
        st.error("❌ Sin conexión a Supabase")
    else:
        st.success(f"✅ Conectado ({len(df)} registros)")

    # --------------------------------------------------------
    # KPIs rápidos del dataset (en memoria, sin generar el PDF)
    # --------------------------------------------------------
    if not df.empty:
        st.markdown("**📊 Resumen del Dataset**")
        try:
            from stats_utils import executive_summary
            _kpi = executive_summary(df, dm.clean_data(df))

            st.metric(
                "Consumo promedio",
                f"${_kpi['consumo_prom']:.2f}",
                help="Promedio de Consumo_Total_MXN en el dataset actual",
            )
            st.metric(
                "Satisfacción promedio",
                f"{_kpi['sat_prom']:.2f} / 5",
                help="Promedio de Satisfaccion_1_al_5 (1=muy baja, 5=muy alta)",
            )
            st.metric(
                "% Hora pico",
                f"{_kpi['pct_pico']:.1f}%",
                help="Porcentaje de registros en horario 8:00-9:00 AM",
            )
            st.metric(
                "Outliers detectados",
                f"{_kpi['outliers']}",
                help=(
                    "Registros descartados por la limpieza del resumen ejecutivo "
                    "(Tiempo_Servicio_Min ≤ 0 o Consumo_Total_MXN > $250). "
                    "Es el mismo criterio que usa el PDF."
                ),
            )
            st.caption(
                f"Última actualización: {datetime.now().strftime('%H:%M')}"
            )
        except Exception as e:
            st.caption(f"⚠️ KPIs no disponibles: {e}")
    else:
        st.info("Sin datos para mostrar KPIs.")

    if st.button("🔄 Recargar datos", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        try:
            from mod_ia_analista import limpiar_cache_ia
            limpiar_cache_ia()
        except Exception:
            pass
        st.rerun()

    # --------------------------------------------------------
    # REPORTES PDF
    # --------------------------------------------------------
    st.divider()
    st.markdown("### 📄 Reportes PDF")

    # Estado del caché IA
    try:
        from mod_ia_analista import stats_cache
        cs = stats_cache()
        if cs["entradas"] > 0:
            st.caption(
                f"🧠 Caché IA: {cs['entradas']} textos | "
                f"hits={cs['hits']} | TTL={cs['ttl_min']}min"
            )
    except Exception:
        pass

    if st.button(
        "📄 Reporte completo",
        use_container_width=True,
        help="11 capítulos, 8 figuras, glosario, análisis IA doble capa.",
    ):
        if df.empty:
            st.warning("Sin datos. Captura o carga registros primero.")
        else:
            with st.spinner("Entrenando modelos y compilando reporte... (~15-20s)"):
                import mod_reporte as mrp
                pdf_bytes, err = mrp.generar_reporte_pdf(df)
            if err:
                st.error(f"Error: {err}")
            else:
                st.session_state["pdf_reporte"] = pdf_bytes
                st.success("✅ Reporte completo generado.")

    if st.button(
        "📋 Resumen 1 página",
        use_container_width=True,
        help="KPIs clave + acciones prioritarias en 1 hoja.",
    ):
        if df.empty:
            st.warning("Sin datos. Captura o carga registros primero.")
        else:
            with st.spinner("Generando resumen..."):
                import mod_reporte as mrp
                pdf_bytes, err = mrp.generar_resumen_pdf(df)
            if err:
                st.error(f"Error: {err}")
            else:
                st.session_state["pdf_resumen"] = pdf_bytes
                st.success("✅ Resumen generado.")

    if "pdf_reporte" in st.session_state:
        st.download_button(
            label="⬇️ Descargar reporte completo",
            data=st.session_state["pdf_reporte"],
            file_name=f"Reporte_Cafeteria_ML_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    if "pdf_resumen" in st.session_state:
        st.download_button(
            label="⬇️ Descargar resumen 1 página",
            data=st.session_state["pdf_resumen"],
            file_name=f"Resumen_Cafeteria_ML_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    if st.button(
        "🧠 Limpiar caché IA",
        use_container_width=True,
        help="Fuerza regeneración de todos los análisis con Gemini.",
    ):
        try:
            from mod_ia_analista import limpiar_cache_ia
            limpiar_cache_ia()
            st.success("Caché IA limpiado.")
        except Exception as e:
            st.error(str(e))

# ============================================================
# BANNER DE ERROR
# ============================================================
if load_error:
    st.error(f"⚠️ Error de conexión: `{load_error}`")
    if st.button("🔄 Reintentar Conexión"):
        st.cache_data.clear()
        st.cache_resource.clear()
        try:
            from mod_ia_analista import limpiar_cache_ia
            limpiar_cache_ia()
        except Exception:
            pass
        st.rerun()
    st.stop()

# ============================================================
# TÍTULO PRINCIPAL
# ============================================================
st.title("☕ Cafetería Universitaria — Laboratorio de ML")
st.markdown(
    """
    **Sistema Integrador de Aprendizaje Automático** aplicado al análisis de clientes
    en horario matutino (7:00 AM – 11:00 AM). Cubre las 4 unidades del temario.
    """
)
st.divider()

# ============================================================
# PESTAÑAS
# ============================================================
tabs = st.tabs([
    "📝 Captura y Datos",
    "🧠 Fundamentos (U1)",
    "📈 Regresión (U2)",
    "🎯 Clasificación (U2)",
    "🔍 No Supervisado (U3)",
    "🤖 Refuerzo (U4)",
])

# ============ 1. CAPTURA Y DATOS ============
with tabs[0]:
    st.subheader("Gestión de Datos")

    tab_manual, tab_file, tab_sintetico = st.tabs(
        ["Formulario Manual", "Subir CSV/Excel", "Generador Sintético"]
    )

    # --- MANUAL ---
    with tab_manual:
        with st.form("form_captura", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                hora = st.number_input("Hora (7.00–11.00)", 7.00, 11.00, 8.00, 0.25)
                espera = st.number_input("Espera (min)", 0.1, 60.0, 3.0, 0.5)
                servicio = st.number_input("Servicio (min)", 0.1, 30.0, 4.0, 0.5)
            with c2:
                tipo = st.selectbox("Tipo de cliente", ["Estudiante", "Profesor", "Administrativo"])
                pago = st.selectbox("Pago", ["(Nulo)", "Efectivo", "Tarjeta", "App"])
                consumo = st.number_input("Consumo (MXN)", 0.0, 500.0, 45.0, 1.0)
            with c3:
                sat = st.selectbox("Satisfacción", ["(Nulo)", 1, 2, 3, 4, 5])
                pico = st.selectbox("¿Hora pico?", ["No", "Sí"])

            if st.form_submit_button("💾 Guardar registro"):
                record = {
                    "ID_Cliente": dm.next_id(df),
                    "Hora_Llegada": float(hora),
                    "Tiempo_Espera_Min": float(espera),
                    "Tiempo_Servicio_Min": float(servicio),
                    "Tipo_Cliente": tipo,
                    "Metodo_Pago": None if pago == "(Nulo)" else pago,
                    "Consumo_Total_MXN": float(consumo),
                    "Satisfaccion_1_al_5": None if sat == "(Nulo)" else int(sat),
                    "Es_Hora_Pico": pico,
                }
                ok, err = dm.save_record(record)
                if ok:
                    st.success("✅ Registro guardado.")
                    st.rerun()
                else:
                    st.error(f"Error: {err}")

    # --- CSV / EXCEL ---
    with tab_file:
        st.markdown("Sube un archivo con las **mismas columnas** que el dataset original.")

        st.markdown("#### 🧪 ¿No tienes un archivo? Genera uno de prueba")
        c_gen1, c_gen2 = st.columns([1, 2])
        with c_gen1:
            n_csv = st.number_input(
                "Cantidad de registros:", min_value=10, max_value=500,
                value=50, step=10, key="gen_csv_n",
            )
        with c_gen2:
            st.write("")
            st.write("")
            if st.button("📄 Generar CSV de Prueba", use_container_width=True):
                df_prueba = dm.generar_datos_sinteticos(n_csv)
                csv_prueba = df_prueba.to_csv(index=False).encode("utf-8")
                st.session_state["csv_prueba"] = csv_prueba
                st.success(f"✅ CSV con {n_csv} registros generado. Descárgalo abajo.")

        if "csv_prueba" in st.session_state:
            st.download_button(
                label="⬇️ Descargar CSV generado",
                data=st.session_state["csv_prueba"],
                file_name="datos_prueba_cafeteria.csv",
                mime="text/csv",
                key="dl_csv_prueba",
            )

        st.divider()

        uploaded_file = st.file_uploader(
            "O sube un archivo CSV o Excel existente", type=["csv", "xlsx"]
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    file_df = pd.read_csv(uploaded_file)
                else:
                    file_df = pd.read_excel(uploaded_file)

                st.write("Vista previa de los datos:")
                st.dataframe(file_df.head(), use_container_width=True)

                if st.button("🚀 Inyectar a Supabase"):
                    missing_cols = [c for c in dm.COLUMNS if c not in file_df.columns]
                    if missing_cols:
                        st.error(f"Faltan columnas: {missing_cols}")
                    else:
                        with st.spinner("Insertando..."):
                            ok, err, count = dm.bulk_insert(file_df)
                        if ok:
                            st.success(f"✅ {count} registros insertados.")
                            st.rerun()
                        else:
                            st.error(f"Error: {err}")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")

    # --- SINTÉTICOS ---
    with tab_sintetico:
        st.markdown("Genera datos sintéticos con la misma distribución para probar tus modelos.")
        n_sint = st.slider("Cantidad de registros a generar:", 10, 500, 50)
        if st.button("🧪 Generar e Insertar Sintéticos"):
            with st.spinner(f"Generando {n_sint} registros..."):
                df_sint = dm.generar_datos_sinteticos(n_sint)
                ok, err, count = dm.bulk_insert(df_sint)
            if ok:
                st.success(f"✅ {count} registros sintéticos insertados.")
                st.rerun()
            else:
                st.error(f"Error: {err}")

    st.divider()
    st.subheader(f"Dataset Actual ({len(df)} registros)")
    if not df.empty:
        st.dataframe(df, use_container_width=True, height=250)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar Dataset", csv, "cafeteria_ml.csv", "text/csv")

# ============ 2. FUNDAMENTOS (U1) ============
with tabs[1]:
    st.subheader("Unidad 1: Introducción al Aprendizaje Automático")
    if df.empty:
        st.info("Sin datos para analizar. Ve a la pestaña de Captura.")
    else:
        u1.mostrar_definicion_mitchell()
        st.divider()
        u1.mostrar_clasificacion_variables(df)
        st.divider()
        u1.mostrar_linea_tiempo()

# ============ 3. REGRESIÓN (U2) ============
with tabs[2]:
    st.subheader("📈 Unidad 2: Regresión — Predicción de Consumo")
    st.markdown(
        "Predice el **Consumo Total (MXN)** usando variables como tiempo de espera, "
        "tipo de cliente y hora de llegada."
    )

    if df.empty:
        st.info("Sin datos. Ve a la pestaña de Captura para generar datos.")
    else:
        if st.button("🚀 Entrenar Modelos de Regresión", key="btn_reg"):
            with st.spinner("Entrenando Regresión Lineal y Árboles..."):
                res_reg = ms.entrenar_regresion(df)

            if res_reg.get("error"):
                st.error(res_reg["error"])
            else:
                st.success("✅ Modelos entrenados correctamente (70% Train / 30% Test).")

                st.markdown("#### 📊 Métricas de Evaluación")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Regresión Lineal**")
                    st.metric("MSE (Error Cuadrático Medio)", f"{res_reg['metricas_lr']['MSE']:.2f}")
                    st.metric("R² (Coef. Determinación)", f"{res_reg['metricas_lr']['R2']:.4f}")
                with c2:
                    st.markdown("**Árbol de Regresión**")
                    st.metric("MSE", f"{res_reg['metricas_dt']['MSE']:.2f}")
                    st.metric("R²", f"{res_reg['metricas_dt']['R2']:.4f}")

                st.pyplot(res_reg["fig"])
        else:
            st.info("Haz clic en el botón para entrenar los modelos.")

# ============ 4. CLASIFICACIÓN (U2) ============
with tabs[3]:
    st.subheader("🎯 Unidad 2: Clasificación — Predicción de Satisfacción")
    st.markdown("Predice si un cliente tendrá **Satisfacción Alta (4-5)** o **Baja (1-3)**.")

    if df.empty:
        st.info("Sin datos para analizar.")
    else:
        modelo_clf = st.selectbox(
            "Selecciona el algoritmo de Clasificación:",
            ["Regresión Logística", "KNN", "SVM", "Árbol de Decisión"],
        )

        if st.button("🚀 Entrenar Clasificador", key="btn_clf"):
            with st.spinner(f"Entrenando {modelo_clf}..."):
                res_clf = ms.entrenar_clasificacion(df, modelo_clf)

            if res_clf.get("error"):
                st.error(res_clf["error"])
            else:
                st.success(f"✅ {modelo_clf} entrenado correctamente.")

                st.markdown("#### 📊 Métricas de Evaluación")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{res_clf['metricas']['Accuracy']:.3f}")
                m2.metric("Precision", f"{res_clf['metricas']['Precision']:.3f}")
                m3.metric("Recall", f"{res_clf['metricas']['Recall']:.3f}")
                m4.metric("F1-Score", f"{res_clf['metricas']['F1']:.3f}")
                if res_clf["metricas"]["AUC"]:
                    m5.metric("AUC-ROC", f"{res_clf['metricas']['AUC']:.3f}")

                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(res_clf["fig_cm"])
                with c2:
                    if res_clf["fig_roc"]:
                        st.pyplot(res_clf["fig_roc"])

                st.markdown("#### 📋 Reporte Detallado por Clase")
                st.dataframe(res_clf["reporte"], use_container_width=True)
        else:
            st.info("Haz clic en el botón para entrenar el clasificador.")

# ============ 5. NO SUPERVISADO (U3) ============
with tabs[4]:
    st.subheader("🔍 Unidad 3: Aprendizaje No Supervisado")

    if df.empty:
        st.info("Sin datos. Ve a la pestaña de Captura para generar datos.")
    else:
        sub_tab1, sub_tab2 = st.tabs(
            ["📏 Distancias entre Clientes", "🎯 Algoritmos de Clustering"]
        )

        # --- 1. DISTANCIAS ---
        with sub_tab1:
            st.markdown(
                "Calcula la **Distancia Euclidiana** (numéricas) y el "
                "**Coeficiente de Jaccard** (categóricas) entre dos clientes."
            )

            if len(df) < 2:
                st.warning("⚠️ Se necesitan al menos 2 registros para calcular distancias.")
            else:
                min_id = int(df["ID_Cliente"].min())
                max_id = int(df["ID_Cliente"].max())

                c1, c2 = st.columns(2)
                with c1:
                    id_a = st.number_input(
                        "ID del Cliente A", min_value=min_id, max_value=max_id,
                        value=min_id, key="dist_id_a",
                    )
                with c2:
                    id_b = st.number_input(
                        "ID del Cliente B", min_value=min_id, max_value=max_id,
                        value=min(max_id, min_id + 1), key="dist_id_b",
                    )

                if st.button("Calcular Distancias"):
                    res_dist = mns.calcular_distancias(df, id_a, id_b)
                    if res_dist.get("error"):
                        st.error(res_dist["error"])
                    else:
                        k1, k2, k3 = st.columns(3)
                        k1.metric("Distancia Euclidiana", f"{res_dist['dist_euclidiana']:.2f}")
                        k2.metric("Similitud de Jaccard", f"{res_dist['jaccard_similitud']:.2%}")
                        k3.metric("Distancia de Jaccard", f"{res_dist['jaccard_distancia']:.2f}")
                        st.info(
                            "💡 Menor distancia euclidiana = más parecidos en comportamiento "
                            "numérico. Jaccard 100% = mismas categorías exactas."
                        )

        # --- 2. CLUSTERING ---
        with sub_tab2:
            alg_cluster = st.selectbox(
                "Selecciona el Algoritmo de Agrupamiento:",
                ["K-Means (Partición)", "Jerárquico (Aglomerativo)", "DBSCAN (Densidad)"],
            )

            if alg_cluster == "K-Means (Partición)":
                k_val = st.slider("Selecciona el número de clústeres (k)", 2, 8, 3)
                if st.button("Ejecutar K-Means"):
                    with st.spinner("Calculando..."):
                        res_km = mns.ejecutar_kmeans(df, n_clusters=k_val)
                    if res_km.get("error"):
                        st.error(res_km["error"])
                    else:
                        st.metric(
                            "Índice de Silhouette", f"{res_km['silhouette']:.4f}",
                            help="Cercano a 1 = mejor separación.",
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            st.pyplot(res_km["fig_codo"])
                        with c2:
                            st.pyplot(res_km["fig_pca"])

            elif alg_cluster == "Jerárquico (Aglomerativo)":
                if st.button("Ejecutar Clustering Jerárquico"):
                    with st.spinner("Construyendo Dendrograma..."):
                        res_jer = mns.ejecutar_jerarquico(df)
                    if res_jer.get("error"):
                        st.error(res_jer["error"])
                    else:
                        st.pyplot(res_jer["fig_dendro"])
                        st.info("💡 El dendrograma muestra cómo los clientes se van fusionando.")

            else:  # DBSCAN
                c1, c2 = st.columns(2)
                with c1:
                    eps_val = st.slider("Radio Epsilon (ε)", 0.1, 2.0, 0.5, 0.1)
                with c2:
                    min_samples_val = st.slider("Mínimo de Puntos (MinPts)", 2, 20, 5)

                if st.button("Ejecutar DBSCAN"):
                    with st.spinner("Detectando densidades..."):
                        res_db = mns.ejecutar_dbscan(df, eps=eps_val, min_samples=min_samples_val)
                    if res_db.get("error"):
                        st.error(res_db["error"])
                    else:
                        m1, m2 = st.columns(2)
                        m1.metric("Clústeres encontrados", res_db["n_clusters"])
                        m2.metric("Puntos de ruido", res_db["n_noise"])
                        st.pyplot(res_db["fig_pca"])
                        st.info("💡 DBSCAN no requiere definir K, pero es sensible al parámetro ε.")

# ============ 6. REFUERZO (U4) ============
with tabs[5]:
    st.subheader("🤖 Unidad 4: Aprendizaje por Refuerzo (Q-Learning)")
    st.markdown(
        """
        **Problema:** Un agente (barista) debe decidir cuántos empleados asignar a la caja
        según la afluencia de clientes (Baja, Media, Alta) para maximizar las ganancias.
        """
    )

    with st.expander("🧮 Ver Ecuación de Bellman", expanded=False):
        st.latex(
            r"Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]"
        )
        st.markdown(
            """
            * **α (alpha):** Tasa de aprendizaje.
            * **γ (gamma):** Factor de descuento.
            * **r:** Recompensa inmediata.
            """
        )

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        episodios = st.slider("Episodios de entrenamiento", 100, 2000, 500, 100)
    with c2:
        alpha = st.slider("Tasa de Aprendizaje (α)", 0.01, 1.0, 0.1, 0.05)
    with c3:
        gamma = st.slider("Factor de Descuento (γ)", 0.1, 1.0, 0.9, 0.05)

    if st.button("🚀 Entrenar Agente (Q-Learning)"):
        with st.spinner(f"Entrenando al agente por {episodios} episodios..."):
            Q, historial = mr.ejecutar_q_learning(episodios=episodios, alpha=alpha, gamma=gamma)
            fig_aprendizaje, fig_q = mr.graficar_resultados(Q, historial)
            politica = mr.obtener_politica_optima(Q)

        st.success("✅ Agente entrenado.")

        st.markdown("### 📈 Evolución del Aprendizaje")
        st.pyplot(fig_aprendizaje)
        st.info(
            "💡 Si la curva azul sube y se estabiliza, el agente ha aprendido una política estable."
        )

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🧠 Matriz Q (Conocimiento Adquirido)")
            st.pyplot(fig_q)
        with c2:
            st.markdown("### 🎯 Política Óptima Aprendida")
            st.dataframe(politica, use_container_width=True, hide_index=True)
    else:
        st.info("Haz clic en el botón para que el agente empiece a aprender.")