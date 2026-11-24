"""Estadísticas descriptivas, resumen ejecutivo y detección de outliers para variables numéricas y categóricas."""
import pandas as pd

NUMERIC_COLS = [
    "Hora_Llegada", "Tiempo_Espera_Min", "Tiempo_Servicio_Min",
    "Consumo_Total_MXN", "Satisfaccion_1_al_5",
]


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estadísticas descriptivas ampliadas:
    n, media, mediana, moda, varianza, desv. std, min, max,
    percentiles 25, 50 y 75.
    """
    rows = []
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        # Moda: puede haber varias; tomamos la primera
        moda_vals = s.mode()
        moda = float(moda_vals.iloc[0]) if not moda_vals.empty else None

        rows.append({
            "Variable": col,
            "n": len(s),
            "Media": round(s.mean(), 3),
            "Mediana": round(s.median(), 3),
            "Moda": round(moda, 3) if moda is not None else None,
            "Varianza": round(s.var(), 3),
            "Desv_Std": round(s.std(), 3),
            "Min": round(s.min(), 3),
            "P25": round(s.quantile(0.25), 3),
            "P50": round(s.quantile(0.50), 3),
            "P75": round(s.quantile(0.75), 3),
            "Max": round(s.max(), 3),
        })
    return pd.DataFrame(rows)


def frequency_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns or df.empty:
        return pd.DataFrame()
    vc = df[col].fillna("(Nulo)").value_counts().reset_index()
    vc.columns = [col, "Frecuencia"]
    vc["Relativa_%"] = (vc["Frecuencia"] / vc["Frecuencia"].sum() * 100).round(2)
    return vc


def executive_summary(df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    """
    Calcula KPIs para el resumen ejecutivo.
    Devuelve un diccionario con valores listos para mostrar en st.metric().
    """
    if df.empty:
        return {
            "n_total": 0, "n_clean": 0, "outliers": 0,
            "consumo_prom": 0.0, "consumo_total": 0.0,
            "espera_prom": 0.0, "espera_max": 0.0,
            "servicio_prom": 0.0,
            "pct_pico": 0.0, "sat_prom": 0.0,
        }

    n_total = len(df)
    n_clean = len(clean_df)
    outliers = n_total - n_clean

    consumo = pd.to_numeric(clean_df["Consumo_Total_MXN"], errors="coerce").dropna()
    espera = pd.to_numeric(df["Tiempo_Espera_Min"], errors="coerce").dropna()
    servicio = pd.to_numeric(clean_df["Tiempo_Servicio_Min"], errors="coerce").dropna()
    sat = pd.to_numeric(clean_df["Satisfaccion_1_al_5"], errors="coerce").dropna()

    pct_pico = 0.0
    if "Es_Hora_Pico" in df.columns and not df.empty:
        pct_pico = (df["Es_Hora_Pico"] == "Sí").mean() * 100

    return {
        "n_total": n_total,
        "n_clean": n_clean,
        "outliers": outliers,
        "consumo_prom": float(consumo.mean()) if not consumo.empty else 0.0,
        "consumo_total": float(consumo.sum()) if not consumo.empty else 0.0,
        "espera_prom": float(espera.mean()) if not espera.empty else 0.0,
        "espera_max": float(espera.max()) if not espera.empty else 0.0,
        "servicio_prom": float(servicio.mean()) if not servicio.empty else 0.0,
        "pct_pico": float(pct_pico),
        "sat_prom": float(sat.mean()) if not sat.empty else 0.0,
    }


def detect_outliers_iqr(df: pd.DataFrame, cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detecta outliers usando el método IQR (1.5 * IQR).

    Devuelve:
      - summary: tabla resumen por variable (Q1, Q3, IQR, límites, n_outliers, %)
      - detail:  filas del DataFrame que contienen al menos un outlier,
                 con una columna 'Outliers en' que lista las variables afectadas.
    """
    if cols is None:
        cols = ["Tiempo_Espera_Min", "Tiempo_Servicio_Min",
                "Consumo_Total_MXN", "Hora_Llegada"]

    summary_rows = []
    outlier_mask = pd.Series(False, index=df.index)
    outlier_vars = pd.Series("", index=df.index)

    for col in cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask_col = ((s < lower) | (s > upper)).fillna(False)

        n_out = int(mask_col.sum())
        pct = (n_out / len(df) * 100) if len(df) else 0.0

        summary_rows.append({
            "Variable": col,
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "IQR": round(iqr, 3),
            "Límite inferior": round(lower, 3),
            "Límite superior": round(upper, 3),
            "N outliers": n_out,
            "% outliers": round(pct, 2),
        })

        outlier_mask = outlier_mask | mask_col
        for idx in df.index[mask_col]:
            prev = outlier_vars.loc[idx]
            outlier_vars.loc[idx] = f"{prev}, {col}" if prev else col

    summary = pd.DataFrame(summary_rows)

    if outlier_mask.any():
        detail = df.loc[outlier_mask].copy()
        detail["Outliers en"] = outlier_vars.loc[outlier_mask].str.lstrip(", ")
    else:
        detail = pd.DataFrame()

    return summary, detail