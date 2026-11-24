"""
Gestión del dataset en Supabase.
"""
import streamlit as st
import pandas as pd
import numpy as np
import random
import math
from supabase import create_client, Client

COLUMNS = [
    "ID_Cliente", "Hora_Llegada", "Tiempo_Espera_Min", "Tiempo_Servicio_Min",
    "Tipo_Cliente", "Metodo_Pago", "Consumo_Total_MXN",
    "Satisfaccion_1_al_5", "Es_Hora_Pico",
]

TO_DB = {
    "ID_Cliente": "id_cliente",
    "Hora_Llegada": "hora_llegada",
    "Tiempo_Espera_Min": "tiempo_espera_min",
    "Tiempo_Servicio_Min": "tiempo_servicio_min",
    "Tipo_Cliente": "tipo_cliente",
    "Metodo_Pago": "metodo_pago",
    "Consumo_Total_MXN": "consumo_total_mxn",
    "Satisfaccion_1_al_5": "satisfaccion_1_al_5",
    "Es_Hora_Pico": "es_hora_pico",
}
FROM_DB = {v: k for k, v in TO_DB.items()}


@st.cache_resource
def _conn() -> Client:
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


def _clean_value(val):
    """Convierte cualquier valor a tipo nativo JSON-safe (elimina NaN, inf, pd.NA)."""
    if val is None:
        return None
    try:
        if val is pd.NA or val is pd.NaT:
            return None
    except Exception:
        pass
    if isinstance(val, (float, np.floating)):
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.bool_,)):
        return bool(val)
    try:
        if pd.isnull(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def load_data() -> tuple[pd.DataFrame, str | None]:
    try:
        res = _conn().table("cafeteria").select("*").execute()
    except Exception as e:
        return pd.DataFrame(columns=COLUMNS), str(e)

    if not res.data:
        return pd.DataFrame(columns=COLUMNS), None

    try:
        df = pd.DataFrame(res.data).rename(columns=FROM_DB)
        if "Satisfaccion_1_al_5" in df.columns:
            df["Satisfaccion_1_al_5"] = pd.to_numeric(df["Satisfaccion_1_al_5"], errors="coerce").astype("Int64")
        if "ID_Cliente" in df.columns:
            df["ID_Cliente"] = pd.to_numeric(df["ID_Cliente"], errors="coerce").astype("Int64")
        if "ID_Cliente" in df.columns and not df.empty:
            df = df.sort_values("ID_Cliente").reset_index(drop=True)
        return df[COLUMNS], None
    except Exception as e:
        return pd.DataFrame(columns=COLUMNS), f"Error procesando datos: {e}"


def save_record(record: dict) -> tuple[bool, str | None]:
    try:
        payload = {TO_DB[k]: _clean_value(record[k]) for k in COLUMNS if k in record}
        _conn().table("cafeteria").insert(payload).execute()
        return True, None
    except Exception as e:
        return False, str(e)


def bulk_insert(df: pd.DataFrame) -> tuple[bool, str | None, int]:
    """Inserta un DataFrame completo en Supabase por lotes de 50, limpiando NaN, forzando tipos y evitando IDs duplicados."""
    if df.empty:
        return False, "El archivo está vacío.", 0
    try:
        # Renombrar columnas al formato de la BD
        db_df = df.rename(columns=TO_DB)
        
        # --- FIX: Forzar tipos correctos ---
        if "satisfaccion_1_al_5" in db_df.columns:
            db_df["satisfaccion_1_al_5"] = pd.to_numeric(
                db_df["satisfaccion_1_al_5"], errors="coerce"
            ).astype("Int64")
        if "id_cliente" in db_df.columns:
            db_df["id_cliente"] = pd.to_numeric(
                db_df["id_cliente"], errors="coerce"
            ).astype("Int64")
        
        # --- FIX: Renumerar IDs para evitar duplicados ---
        current_df, _ = load_data()
        existing_ids = set(current_df["ID_Cliente"].dropna().astype(int).tolist()) if not current_df.empty else set()
        next_available_id = (max(existing_ids) + 1) if existing_ids else 1
        
        # Limpiar y renumerar cada registro
        records = []
        for _, row in db_df.iterrows():
            record = {col: _clean_value(row[col]) for col in db_df.columns}
            
            # Si el ID ya existe, asignar uno nuevo
            if record.get("id_cliente") is not None and int(record["id_cliente"]) in existing_ids:
                record["id_cliente"] = next_available_id
                next_available_id += 1
            elif record.get("id_cliente") is not None:
                existing_ids.add(int(record["id_cliente"]))
                next_available_id = max(next_available_id, int(record["id_cliente"]) + 1)
            
            records.append(record)
        
        count = 0
        for i in range(0, len(records), 50):
            batch = records[i:i+50]
            _conn().table("cafeteria").insert(batch).execute()
            count += len(batch)
        return True, None, count
    except Exception as e:
        return False, str(e), 0


def generar_datos_sinteticos(n: int) -> pd.DataFrame:
    random.seed(42)
    tipos = ["Estudiante", "Profesor", "Administrativo"]
    pesos_tipo = [0.7, 0.2, 0.1]
    pagos = ["Efectivo", "Tarjeta", "App", None]
    pesos_pago = [0.30, 0.35, 0.25, 0.10]

    current_df, _ = load_data()
    start_id = next_id(current_df) if not current_df.empty else 1

    registros = []
    for i in range(start_id, start_id + n):
        tipo = random.choices(tipos, weights=pesos_tipo)[0]
        hora = round(random.triangular(7.0, 11.0, 8.5), 2)
        espera = round(min(max(random.expovariate(1/4), 0.5), 25.0), 1)
        servicio = round(max(random.gauss(4, 1.5), 0.5), 1)
        base_consumo = {"Estudiante": 45, "Profesor": 70, "Administrativo": 55}[tipo]
        consumo = round(max(random.gauss(base_consumo, 15), 5.0), 1)
        sat = random.choices([1, 2, 3, 4, 5, None], weights=[0.05, 0.10, 0.20, 0.35, 0.25, 0.05])[0]

        registros.append({
            "ID_Cliente": i,
            "Hora_Llegada": hora,
            "Tiempo_Espera_Min": espera,
            "Tiempo_Servicio_Min": servicio,
            "Tipo_Cliente": tipo,
            "Metodo_Pago": random.choices(pagos, weights=pesos_pago)[0],
            "Consumo_Total_MXN": consumo,
            "Satisfaccion_1_al_5": sat,
            "Es_Hora_Pico": "Sí" if 8.0 <= hora <= 9.0 else "No",
        })

    df = pd.DataFrame(registros)
    # FORZAR Int64 para que nulos sean pd.NA y no NaN
    df["Satisfaccion_1_al_5"] = df["Satisfaccion_1_al_5"].astype("Int64")
    df["ID_Cliente"] = df["ID_Cliente"].astype("Int64")
    return df


def next_id(df: pd.DataFrame) -> int:
    if df.empty or df["ID_Cliente"].dropna().empty:
        return 1
    return int(df["ID_Cliente"].max()) + 1


def validate_record(record: dict) -> list[str]:
    errors = []
    h = record["Hora_Llegada"]
    if not (7.00 <= h <= 11.00):
        errors.append("Hora_Llegada debe estar entre 7.00 y 11.00.")
    if record["Tiempo_Espera_Min"] <= 0:
        errors.append("Tiempo_Espera_Min debe ser > 0.")
    if record["Tiempo_Servicio_Min"] <= 0:
        errors.append("Tiempo_Servicio_Min debe ser > 0.")
    if record["Tipo_Cliente"] not in ("Estudiante", "Profesor", "Administrativo"):
        errors.append("Tipo_Cliente inválido.")
    if record["Metodo_Pago"] not in (None, "", "Efectivo", "Tarjeta", "App"):
        errors.append("Metodo_Pago inválido.")
    if record["Consumo_Total_MXN"] < 0:
        errors.append("Consumo_Total_MXN debe ser >= 0.")
    sat = record["Satisfaccion_1_al_5"]
    if sat is not None and sat != "" and sat not in (1, 2, 3, 4, 5):
        errors.append("Satisfaccion_1_al_5 debe estar entre 1 y 5.")
    if record["Es_Hora_Pico"] not in ("Sí", "No"):
        errors.append("Es_Hora_Pico debe ser 'Sí' o 'No'.")
    return errors


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    clean = df.copy()
    clean = clean[clean["Tiempo_Servicio_Min"] > 0]
    clean = clean[clean["Consumo_Total_MXN"] <= 250]
    return clean.reset_index(drop=True)