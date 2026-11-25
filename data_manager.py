"""
Gestión del dataset en Supabase.
"""
import streamlit as st
import pandas as pd
import numpy as np
import random
import math
from supabase import create_client, Client

# ============================================================
# LÍMITE DE REGISTROS Y NORMALIZACIÓN
# ============================================================
MAX_REGISTROS = 200
NORMALIZAR_MIN = 1
NORMALIZAR_MAX = MAX_REGISTROS


def count_records() -> int:
    """Conteo exacto desde Supabase. -1 si falla la conexión."""
    try:
        res = (
            _conn()
            .table("cafeteria")
            .select("id_cliente", count="exact")
            .limit(1)
            .execute()
        )
        return int(res.count or 0)
    except Exception:
        return -1


def slots_disponibles() -> int:
    """Cuántos registros caben antes de llegar a MAX_REGISTROS. -1 si falla."""
    n = count_records()
    if n < 0:
        return -1
    return max(0, MAX_REGISTROS - n)


def normalizar_a(n_objetivo: int) -> tuple[bool, str | None, int, int, int, int]:
    """
    Ajusta la BD a exactamente n_objetivo registros (1 <= n <= 200).
    - Si hay M > n_objetivo: borra los M - n_objetivo MÁS RECIENTES (ID alto),
      conserva los n_objetivo MÁS ANTIGUOS (ID bajo).
    - Si hay M < n_objetivo: rellena con n_objetivo - M sintéticos (seed=42).
    - Si hay M == n_objetivo: no hace nada.
    Retorna (ok, err, antes, despues, borrados, agregados).
    """
    try:
        n_objetivo = int(n_objetivo)
    except (TypeError, ValueError):
        return False, f"Valor inválido: {n_objetivo!r}", 0, 0, 0, 0
    if n_objetivo < NORMALIZAR_MIN or n_objetivo > NORMALIZAR_MAX:
        return (
            False,
            f"Valor fuera de rango [{NORMALIZAR_MIN}, {NORMALIZAR_MAX}]: {n_objetivo}",
            0, 0, 0, 0,
        )

    antes = count_records()
    if antes < 0:
        return False, "No se pudo leer el conteo actual.", 0, 0, 0, 0

    # Caso exacto
    if antes == n_objetivo:
        return True, None, antes, antes, 0, 0

    # Caso reducir: borrar los más recientes (ID alto), conservar los más antiguos
    if antes > n_objetivo:
        df_actual, err_load = load_data()
        if df_actual.empty:
            return False, f"No se pudo leer la BD: {err_load}", antes, antes, 0, 0
        df_ordenado = df_actual.sort_values("ID_Cliente").reset_index(drop=True)
        ids_borrar = [
            int(x) for x in df_ordenado.iloc[n_objetivo:]["ID_Cliente"].tolist()
        ]
        if not ids_borrar:
            return True, None, antes, antes, 0, 0
        borrados = 0
        try:
            # Borrado en lotes de 100 para no exceder límites de Supabase
            for i in range(0, len(ids_borrar), 100):
                lote = ids_borrar[i:i + 100]
                _conn().table("cafeteria").delete().in_("id_cliente", lote).execute()
                borrados += len(lote)
        except Exception as e:
            fallo = count_records()
            return (
                False,
                f"Error al borrar ({borrados} eliminados antes del fallo): {e}",
                antes,
                fallo if fallo >= 0 else antes - borrados,
                borrados,
                0,
            )
        despues = count_records()
        if despues < 0:
            despues = antes - borrados
        return True, None, antes, despues, borrados, 0

    # Caso rellenar: M < n_objetivo -> agregar sintéticos
    faltan = n_objetivo - antes
    df_relleno = generar_datos_sinteticos(faltan)
    ok, err, count = bulk_insert(df_relleno, respetar_limite=False)
    if not ok:
        return False, f"Error al rellenar: {err}", antes, antes, 0, 0
    despues = count_records()
    if despues < 0:
        despues = antes + count
    return True, None, antes, despues, 0, count


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


PAGE_SIZE = 1000  # tamaño de bloque de lectura (PostgREST capa a db-max-rows)
_MAX_ROWS_LEIDOS = 10_000_000  # salvaguarda anti bucle infinito


def _max_id_db() -> int:
    """ID máximo real de Supabase consultando UNA sola fila.

    `order(id_cliente, desc).limit(1)` es inmune al `db-max-rows` de
    PostgREST que recorta los SELECT largos (1000 filas por defecto).
    Devuelve 0 si la tabla está vacía o si falla la BD.
    """
    try:
        res = (
            _conn()
            .table("cafeteria")
            .select("id_cliente")
            .order("id_cliente", desc=True)
            .limit(1)
            .execute()
        )
        if res.data:
            return int(res.data[0]["id_cliente"])
    except Exception:
        pass
    return 0


def load_data() -> tuple[pd.DataFrame, str | None]:
    """Lee TODOS los registros de Supabase, paginando por bloques.

    PostgREST recorta cada SELECT a `db-max-rows` filas (1000 por defecto) sin
    avisar, así que se pide con .range() y se avanza por lo realmente recibido:
    el conteo exacto (count_records) marca la meta y, si la BD no lo da, se
    corta en el primer bloque corto o vacío.
    """
    rows: list[dict] = []
    try:
        total = count_records()
        start = 0
        page_max = 0  # página más grande recibida (cap real del servidor)
        while True:
            res = (
                _conn()
                .table("cafeteria")
                .select("*")
                .order("id_cliente")
                .range(start, start + PAGE_SIZE - 1)
                .execute()
            )
            batch = res.data or []
            if not batch:
                break
            rows.extend(batch)
            start += len(batch)
            page_max = max(page_max, len(batch))
            if total >= 0:
                if start >= total:
                    break
            elif len(batch) < page_max:
                break  # bloque corto = última página (sin conteo fiable)
            if start >= _MAX_ROWS_LEIDOS:
                break
    except Exception as e:
        return pd.DataFrame(columns=COLUMNS), str(e)

    if not rows:
        return pd.DataFrame(columns=COLUMNS), None

    try:
        df = pd.DataFrame(rows).rename(columns=FROM_DB)
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


def bulk_insert(
    df: pd.DataFrame, respetar_limite: bool = False
) -> tuple[bool, str | None, int]:
    """Inserta un DataFrame en Supabase forzando IDs nuevos (evita duplicados siempre).

    La ingesta es libre (sin tope de registros): el único límite del sistema es
    el Normalizador de BD, que deja la base en 1..MAX_REGISTROS. El parámetro
    respetar_limite se conserva por compatibilidad, pero ya no se usa por defecto.
    """
    if df.empty:
        return False, "El archivo está vacío.", 0
    if respetar_limite:
        libres = slots_disponibles()
        if libres < 0:
            return False, "No se pudo verificar el conteo actual en Supabase.", 0
        if len(df) > libres:
            return False, (
                f"El límite es {MAX_REGISTROS} registros y solo caben {libres} más "
                f"(la base tiene {MAX_REGISTROS - libres}). Se intentaron insertar "
                f"{len(df)}. Usa el Normalizador de BD para ajustar el tamaño."
            ), 0
    try:
        db_df = df.rename(columns=TO_DB)
        
        # Forzar tipos correctos
        if "satisfaccion_1_al_5" in db_df.columns:
            db_df["satisfaccion_1_al_5"] = pd.to_numeric(
                db_df["satisfaccion_1_al_5"], errors="coerce"
            ).astype("Int64")
        
        # Siguiente ID desde la BD real: 1 fila, inmune al cap de 1000 de
        # PostgREST. Antes se derivaba de load_data(), que devolvía solo 1000
        # filas y provocaba colisiones de PK (duplicate key) al insertar más.
        existing_max = _max_id_db()
        next_available_id = existing_max + 1
        
        # FORZAR: todos los registros reciben un ID nuevo (ignoramos el ID del CSV)
        records = []
        for _, row in db_df.iterrows():
            record = {col: _clean_value(row[col]) for col in db_df.columns if col != "id_cliente"}
            record["id_cliente"] = next_available_id
            next_available_id += 1
            records.append(record)
        
        # Insertar por lotes de 50
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

    # El ID inicial ya no depende de load_data() (que antes venía capado):
    # se consulta el máximo real de la BD. bulk_insert reasigna los IDs.
    start_id = _max_id_db() + 1

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