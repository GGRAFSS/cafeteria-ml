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
import os

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
# ESTILO INSTITUCIONAL (CSS v5.1)
# ============================================================
CSS = """
:root{
    --azul:#003366;
    --azul-oscuro:#0b2545;
    --dorado:#C9A961;
    --gris:#F5F5F5;
}
.stApp{ background:linear-gradient(180deg,#FFFFFF 0%,#FAFBFC 100%); }
h1, h2, h3{ color:var(--azul); }

/* ---------- HERO ---------- */
.hero{
    position:relative; min-height:240px; margin:0 0 1rem 0; padding:2.3rem 2.6rem;
    border-radius:16px; overflow:hidden;
    display:flex; flex-direction:column; justify-content:center;
    background-image:
        linear-gradient(105deg, rgba(0,51,102,.94) 0%, rgba(0,51,102,.78) 48%, rgba(11,37,69,.42) 100%),
        url("/app/static/hero.jpeg");
    background-size:cover; background-position:center;
    box-shadow:0 10px 26px rgba(0,51,102,.28);
}
.hero h1, .hero h3, .hero p{
    position:relative; margin:0; color:#FFFFFF !important;
    text-shadow:0 2px 8px rgba(0,0,0,.55);
}
.hero h1{ font-size:2.1rem; font-weight:800; letter-spacing:.3px; }
.hero h3{ font-size:1.12rem; font-weight:600; color:var(--dorado) !important; margin:.4rem 0 .5rem 0; }
.hero p{ font-size:.95rem; opacity:.95; }

/* ---------- BOTONES ---------- */
.stButton > button{ border-radius:9px; font-weight:600; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"]{
    background:var(--azul) !important; color:#FFFFFF !important;
    border:1px solid var(--azul) !important;
    box-shadow:0 4px 14px rgba(0,51,102,.22);
}
.stButton > button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p{ color:#FFFFFF !important; }
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover{
    background:var(--dorado) !important; color:var(--azul-oscuro) !important;
    border-color:#b3924f !important;
}
.stButton > button[kind="primary"]:hover p, button[data-testid="stBaseButton-primary"]:hover p{
    color:var(--azul-oscuro) !important;
}

/* ---------- PESTAÑAS ---------- */
.stTabs [data-baseweb="tab-list"]{
    gap:6px; background:var(--gris); padding:6px;
    border-radius:12px; border:1px solid #E3E6EB;
}
.stTabs [data-baseweb="tab"]{ border-radius:9px; padding:6px 14px; font-weight:600; color:#33475B; background:transparent; }
.stTabs [aria-selected="true"]{ background:var(--azul) !important; }
.stTabs [aria-selected="true"] p{ color:#FFFFFF !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }

/* ---------- MÉTRICAS ---------- */
[data-testid="stMetric"]{
    background:var(--gris); border-left:4px solid var(--azul);
    border-radius:10px; padding:.6rem .9rem;
}
[data-testid="stMetricLabel"] p{ color:#33475B !important; font-weight:600; }
[data-testid="stMetricValue"]{ color:var(--azul); }

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"]{ background:var(--azul-oscuro); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3{ color:#FFFFFF; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{ color:#E8EEF7; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{ color:#BFD0E4; }
[data-testid="stSidebar"] hr{ border-color:rgba(255,255,255,.18); }
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button{
    background:var(--gris) !important; color:var(--azul) !important;
    border:1px solid rgba(201,169,97,.55) !important;
}
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stDownloadButton > button p{ color:var(--azul) !important; }
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stDownloadButton > button:hover{
    background:var(--dorado) !important; color:var(--azul-oscuro) !important;
    border-color:var(--dorado) !important;
}
[data-testid="stSidebar"] .stButton > button:hover p,
[data-testid="stSidebar"] .stDownloadButton > button:hover p{ color:var(--azul-oscuro) !important; }
[data-testid="stSidebar"] [data-testid="stAlert"] p{ color:#1A1A1A !important; }
[data-testid="stSidebar"] [data-testid="stMetric"]{
    background:rgba(255,255,255,.10); border-left:4px solid var(--dorado);
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p{ color:#CFDCEC !important; }
[data-testid="stSidebar"] [data-testid="stMetricValue"]{ color:#FFFFFF; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def _toast_diferido(mensaje: str, icono: str = "✅") -> None:
    """Encola un toast para mostrarlo después del siguiente ``st.rerun()``."""
    st.session_state["_toast_pendiente"] = (mensaje, icono)


_pendiente = st.session_state.pop("_toast_pendiente", None)
if _pendiente:
    st.toast(_pendiente[0], icon=_pendiente[1])

# ============================================================
# ============================================================
# POPUP DE BIENVENIDA (1 vez por sesión)
# ============================================================
@st.dialog("👋 Bienvenido al Laboratorio de ML", width="large")
def _mostrar_bienvenida() -> None:
    """Popup institucional de bienvenida (1 vez por sesión o desde el sidebar)."""
    c_logo1, c_logo2 = st.columns(2)
    if os.path.exists("assets/logo_tecnm_virtual.png"):
        with c_logo1:
            st.image("assets/logo_tecnm_virtual.png", width=110)
    if os.path.exists("assets/logo_maestria_ia.png"):
        with c_logo2:
            st.image("assets/logo_maestria_ia.png", width=95)

    st.markdown("### ☕ Cafetería Universitaria — Laboratorio de ML")
    st.markdown(
        """
        **Materia:** Aprendizaje Automático · **Posgrado:** Maestría en Inteligencia Artificial  
        **Institución:** TecNM — Instituto Tecnológico de Tijuana · **Periodo:** 2026

        **Autores:** Rodríguez Avilez Selene · Raygoza Flores Gabriel  
        **Docente:** Dra. Claudia Ibeth González Berrelleza
        """
    )
    st.markdown(
        """
        **¿Qué puedes hacer en esta app?**
        - 📝 **Capturar, cargar y normalizar** los registros de clientes (7:00–11:00 AM).
        - 🧠 **Entrenar y comparar** regresión, clasificación, clustering (K-Means,
          jerárquico, DBSCAN) y refuerzo (Q-Learning).
        - 📄 **Descargar reportes PDF** ejecutivos y didácticos con métricas y figuras.
        """
    )
    if st.button("Comenzar 🚀", type="primary", use_container_width=True):
        st.session_state["_bienvenida_mostrada"] = True
        st.rerun()


if not st.session_state.get("_bienvenida_mostrada"):
    _mostrar_bienvenida()

# ============================================================
# CARGA DE DATOS
# ============================================================
df, load_error = dm.load_data()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    if os.path.exists("assets/logo_tecnm_virtual.png"):
        st.image("assets/logo_tecnm_virtual.png", width=90)
    if os.path.exists("assets/logo_maestria_ia.png"):
        st.image("assets/logo_maestria_ia.png", width=80)
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
        type="primary",
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
                st.toast("📄 Reporte completo generado.", icon="✅")

    if st.button(
        "📋 Resumen 1 página",
        type="primary",
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
                st.toast("📋 Resumen de 1 página generado.", icon="✅")

    if st.button(
        "🎓 Reporte didáctico",
        type="primary",
        use_container_width=True,
        help="Explica cada figura y tabla en 7 bloques: qué se ve, interpretación "
             "técnica, palabras simples, temario, pedagogía, errores comunes y "
             "qué técnicas NO aplican.",
    ):
        if df.empty:
            st.warning("Sin datos. Captura o carga registros primero.")
        else:
            with st.spinner("Generando reporte didáctico... (~15-20s)"):
                import mod_reporte as mrp
                pdf_bytes, err = mrp.generar_reporte_didactico(df)
            if err:
                st.error(f"Error: {err}")
            else:
                st.session_state["pdf_didactico"] = pdf_bytes
                st.toast("🎓 Reporte didáctico generado.", icon="✅")

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

    if "pdf_didactico" in st.session_state:
        st.download_button(
            label="⬇️ Descargar reporte didáctico",
            data=st.session_state["pdf_didactico"],
            file_name=f"Reporte_Didactico_{datetime.now().strftime('%Y%m%d')}.pdf",
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
            st.toast("🧠 Caché IA limpiado.", icon="✅")
        except Exception as e:
            st.error(str(e))

    st.divider()
    if st.button("ℹ️ Acerca del proyecto", use_container_width=True):
        st.session_state["_bienvenida_mostrada"] = False
        st.rerun()

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
# HERO INSTITUCIONAL
# ============================================================
st.markdown(
    """
    <div class="hero hero-container">
        <h1>☕ Cafetería Universitaria</h1>
        <h3>Laboratorio de Aprendizaje Automático</h3>
        <p>Maestría en IA · TecNM · Instituto Tecnológico de Tijuana</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ============================================================
# PESTAÑAS
# ============================================================
tabs = st.tabs([
    "📝 Captura",
    "🧠 U1 · Fundamentos",
    "📈 U2 · Regresión",
    "🎯 U2 · Clasificación",
    "🔍 U3 · Clustering",
    "🤖 U4 · Refuerzo",
])

# ============ 1. CAPTURA Y DATOS ============
with tabs[0]:
    st.subheader("Gestión de Datos")

    # Conteo en vivo de la base: solo informativo, la ingesta es libre.
    n_actual = dm.count_records()
    if n_actual < 0:
        st.warning(
            "No se pudo leer el conteo de registros en Supabase: la ingesta "
            "sigue disponible, pero el conteo actual es desconocido."
        )
    else:
        st.caption(f"📦 Base actual: **{n_actual}** registros")

    tab_manual, tab_file, tab_sintetico, tab_norm = st.tabs(
        [
            "Formulario Manual",
            "Subir CSV/Excel",
            "Generador Sintético",
            "⚖️ Normalizador de BD",
        ]
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
                    _toast_diferido("💾 Registro guardado.", icono="✅")
                    st.rerun()
                else:
                    st.error(f"Error: {err}")

    # --- CSV / EXCEL ---
    with tab_file:
        st.markdown("Sube un archivo con las **mismas columnas** que el dataset original.")

        st.markdown("#### 🧪 ¿No tienes un archivo? Genera uno de prueba")
        _max_csv = 10000
        c_gen1, c_gen2 = st.columns([1, 2])
        with c_gen1:
            n_csv = st.number_input(
                "Cantidad de registros:", min_value=1, max_value=_max_csv,
                value=min(50, _max_csv), step=1, key="gen_csv_n",
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
                st.caption(f"Filas en el archivo: **{len(file_df)}**")

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
        _max_sint = 5000
        n_sint = st.slider(
            "Cantidad de registros a generar:", 1, _max_sint, min(50, _max_sint)
        )
        if st.button(
            "🧪 Generar e Insertar Sintéticos",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(f"Generando {n_sint} registros..."):
                df_sint = dm.generar_datos_sinteticos(n_sint)
                ok, err, count = dm.bulk_insert(df_sint)
            if ok:
                _toast_diferido(f"🧪 {count} registros sintéticos insertados.")
                st.rerun()
            else:
                st.error(f"Error: {err}")

    # --- NORMALIZADOR DE BD ---
    with tab_norm:
        st.markdown(
            "Deja la base en **exactamente N registros** (1–200). Si hay más, se "
            "eliminan los **más recientes** (ID alto) y se conservan los más "
            "antiguos; si hay menos, se completan con datos sintéticos (seed 42)."
        )
        if n_actual < 0:
            st.error(
                "No se pudo leer el conteo actual de la base: el normalizador no "
                "está disponible en este momento."
            )
        else:
            _msg_norm = st.session_state.pop("norm_ok", None)
            if _msg_norm:
                st.toast(_msg_norm.removeprefix("✅ ").strip(), icon="⚖️")
            st.caption(
                f"📦 Base actual: **{n_actual} / {dm.MAX_REGISTROS}** registros"
            )
            n_objetivo = st.slider(
                "Registros objetivo:",
                min_value=dm.NORMALIZAR_MIN,
                max_value=dm.NORMALIZAR_MAX,
                value=min(max(n_actual, dm.NORMALIZAR_MIN), dm.NORMALIZAR_MAX),
                step=1,
                key="norm_objetivo",
            )
            delta = n_objetivo - n_actual
            if delta == 0:
                st.info("La base ya está en ese tamaño: no se hará ningún cambio.")
            elif delta < 0:
                st.warning(
                    f"Se eliminarán **{abs(delta)}** registros (los más recientes, "
                    "ID más alto)."
                )
            else:
                st.warning(
                    f"Se agregarán **{delta}** registros sintéticos para llegar "
                    "al objetivo."
                )
            confirmar = st.checkbox(
                "Confirmo que quiero ajustar la base a "
                f"**{n_objetivo}** registros (esta acción modifica datos).",
                key="norm_confirm",
            )
            if st.button(
                "⚖️ Normalizar base de datos",
                type="primary",
                key="btn_normalizar",
                disabled=not confirmar,
            ):
                with st.spinner(f"Ajustando la base a {n_objetivo} registros..."):
                    ok, err, antes, despues, borrados, agregados = dm.normalizar_a(
                        n_objetivo
                    )
                if ok:
                    st.session_state["norm_ok"] = (
                        f"✅ Base normalizada: {antes} → {despues} registros "
                        f"({borrados} eliminados, {agregados} agregados)."
                    )
                    st.cache_data.clear()
                    try:
                        from mod_ia_analista import limpiar_cache_ia
                        limpiar_cache_ia()
                    except Exception:
                        pass
                    st.rerun()
                else:
                    st.error(f"No se pudo normalizar: {err}")

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
        if st.button(
            "🚀 Entrenar Modelos de Regresión",
            key="btn_reg",
            type="primary",
        ):
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

        if st.button(
            "🚀 Entrenar Clasificador",
            key="btn_clf",
            type="primary",
        ):
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