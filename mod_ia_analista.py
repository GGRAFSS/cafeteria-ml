"""
Módulo Analista con IA (Gemini)
Genera interpretaciones dinámicas de métricas, gráficas y conclusiones
a partir de los valores actuales de la base de datos.

v5.0:
- Tercer tipo `didactico`: devuelve los siete campos de CAMPOS_DIDACTICOS
  (qué se ve, interpretación, palabras simples, temario, pedagogía,
  errores frecuentes y técnicas que NO aplican), base del Reporte
  Didáctico de mod_reporte.

v3.3:
- Parámetro `tipo` para distinguir "analisis" (doble capa) vs "decisiones"
  (lista operativa corta).
- Caché en memoria thread-safe (TTL 1h).
- Compatible con ThreadPoolExecutor.
- Parámetro `instruccion_extra` para sesgar el enfoque por capítulo.
"""
import json
import os
import time
import hashlib
from threading import Lock
import streamlit as st

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


# ============================================================
# CAMPOS DEL REPORTE DIDÁCTICO (v5.0)
# ============================================================
# Orden oficial de los siete bloques de cada ficha didáctica.
CAMPOS_DIDACTICOS = (
    ("que_se_ve", "Qué se ve"),
    ("tecnica", "Interpretación técnica"),
    ("simple", "En palabras simples"),
    ("temario", "Relación con el temario"),
    ("pedagogia", "Por qué importa pedagógicamente"),
    ("errores", "Errores comunes al interpretar"),
    ("no_aplica", "¿Qué técnicas NO se aplican aquí y por qué?"),
)

PLACEHOLDER_DIDACTICO = (
    "[Fase 5B] Contenido pendiente: este bloque se completará con el análisis "
    "didáctico generado por IA a partir de los datos reales del reporte."
)


def _normalizar_campos_didacticos(campos: dict | None, texto_base: str = "") -> dict:
    """Completa las siete claves de una ficha didáctica sin dejar huecos.

    Acepta tanto las claves canónicas como las etiquetas legibles y garantiza
    que siempre exista texto, de modo que el PDF nunca dibuje un bloque vacío.
    """
    campos = campos or {}
    base = (texto_base or "").strip() or PLACEHOLDER_DIDACTICO
    salida = {}
    for clave, etiqueta in CAMPOS_DIDACTICOS:
        valor = campos.get(clave, campos.get(etiqueta))
        valor = str(valor).strip() if valor is not None else ""
        salida[clave] = valor or base
    return salida


def _parse_campos_didacticos(raw: str, fallback: dict) -> dict:
    """Interpreta la respuesta JSON del modelo; ante cualquier fallo usa
    `fallback` (siete claves ya normalizadas)."""
    texto = (raw or "").strip()
    if "```" in texto:
        trozos = [t.strip() for t in texto.split("```") if t.strip()]
        if trozos:
            texto = max(trozos, key=len)
            if texto.lower().startswith("json"):
                texto = texto[4:].strip()
    ini, fin = texto.find("{"), texto.rfind("}")
    if ini == -1 or fin <= ini:
        return fallback
    try:
        datos = json.loads(texto[ini:fin + 1])
    except Exception:
        return fallback
    if not isinstance(datos, dict):
        return fallback
    salida = dict(fallback)
    for clave, etiqueta in CAMPOS_DIDACTICOS:
        valor = datos.get(clave, datos.get(etiqueta))
        if isinstance(valor, (list, tuple)):
            valor = " • ".join(str(v).strip() for v in valor if str(v).strip())
        if valor is None:
            continue
        valor = str(valor).replace("\r\n", "\n").strip()
        if valor:
            salida[clave] = valor
    return salida




# ============================================================
# CACHÉ EN MEMORIA (thread-safe, TTL)
# ============================================================
_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_LOCK = Lock()
_CACHE_TTL = 3600
_CACHE_HITS = 0
_CACHE_MISSES = 0


def _get_api_key() -> str | None:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            val = str(st.secrets["GEMINI_API_KEY"]).strip()
            if val:
                return val
    except Exception:
        pass
    return (os.getenv("GEMINI_API_KEY", "").strip() or None)


def _limpiar_texto_pdf(texto: str) -> str:
    limpio = texto.replace("\r\n", "\n").strip()
    for token in ("```", "###", "##", "# "):
        limpio = limpio.replace(token, "")
    return limpio.strip()


def _cache_key(seccion, datos_metricas, fallback_texto, instruccion_extra, img_bytes):
    h = hashlib.sha256()
    h.update(seccion.encode("utf-8"))
    h.update(json.dumps(datos_metricas, sort_keys=True, default=str).encode("utf-8"))
    h.update(fallback_texto.encode("utf-8"))
    h.update((instruccion_extra or "").encode("utf-8"))
    if img_bytes:
        h.update(img_bytes)
    return h.hexdigest()


def _cache_get(key: str) -> str | None:
    global _CACHE_HITS
    with _CACHE_LOCK:
        if key in _CACHE:
            texto, ts = _CACHE[key]
            if (time.time() - ts) < _CACHE_TTL:
                _CACHE_HITS += 1
                return texto
            del _CACHE[key]
    return None


def _cache_set(key: str, texto: str) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (texto, time.time())


def limpiar_cache_ia() -> None:
    global _CACHE_HITS, _CACHE_MISSES
    with _CACHE_LOCK:
        _CACHE.clear()
        _CACHE_HITS = 0
        _CACHE_MISSES = 0


def stats_cache() -> dict:
    with _CACHE_LOCK:
        return {
            "entradas": len(_CACHE),
            "hits": _CACHE_HITS,
            "misses": _CACHE_MISSES,
            "ttl_min": _CACHE_TTL // 60,
        }


# ============================================================
# ANÁLISIS CON IA
# ============================================================
def analizar_seccion_ia(
    seccion: str,
    datos_metricas: dict,
    fallback_texto: str,
    imagen_png_bytes: bytes | None = None,
    instruccion_extra: str | None = None,
    doble_capa: bool = True,
    tipo: str = "analisis",   # "analisis" | "decisiones" | "didactico"
    fallback_campos: dict | None = None,
) -> dict:
    """
    Retorna siempre: {"tecnico": str, "simple": str | None}
    - tipo="analisis": "tecnico" = análisis formal; "simple" = versión coloquial.
    - tipo="decisiones": "tecnico" = lista de decisiones (una por línea, con '•');
      "simple" = None.
    - tipo="didactico": devuelve las siete claves de CAMPOS_DIDACTICOS
      (que_se_ve, tecnica, simple, temario, pedagogia, errores, no_aplica);
      `fallback_campos` permite dar un texto de respaldo por clave.
    """
    global _CACHE_MISSES

    key = _cache_key(seccion, datos_metricas, fallback_texto,
                     instruccion_extra, imagen_png_bytes)
    key = f"{key}:doble={doble_capa}:tipo={tipo}"
    cached = _cache_get(key)
    if cached is not None:
        try:
            return json.loads(cached)
        except Exception:
            pass

    api_key = _get_api_key()
    if tipo == "didactico":
        fallback = _normalizar_campos_didacticos(fallback_campos, fallback_texto)
    else:
        fallback = {"tecnico": fallback_texto, "simple": None}
    if not HAS_GENAI or not api_key:
        return fallback

    extra = (
        f"\nEnfoque específico solicitado:\n{instruccion_extra}\n"
        if instruccion_extra else ""
    )

    if tipo == "decisiones":
        formato = (
            "Devuelve SOLO una lista de 3 o 4 decisiones OPERATIVAS concretas, "
            "una por línea, cada una empezando con el carácter '•' seguido de un espacio. "
            "Sin párrafos, sin títulos, sin explicaciones, sin markdown, sin numeración. "
            "Cada decisión debe responder qué hacer, dónde y cuándo. "
            "Máximo 18 palabras por decisión. "
            "Prohibido: hiperparámetros, siglas técnicas, referencias a código, "
            "nombres como 'R²', 'F1', 'AUC', 'silhouette', 'MSE'. "
            "Traduce todo a lenguaje operativo de cafetería (empleados, caja, horarios, mesas, promociones)."
        )
    elif tipo == "didactico":
        formato = (
            "Devuelve EXCLUSIVAMENTE un objeto JSON válido, sin texto antes ni "
            "después y sin bloques de código, con estas siete claves exactas:\n"
            '"que_se_ve": descripción visual objetiva de la figura o tabla '
            "(tipo de gráfico, ejes, escalas, colores, categorías y rangos "
            "observados), 60-90 palabras.\n"
            '"tecnica": interpretación técnica usando SOLO los valores '
            "proporcionados (métricas, promedios, proporciones, tamaños de "
            "muestra), 70-110 palabras.\n"
            '"simple": la misma conclusión sin jerga y con una analogía '
            "cotidiana de cafetería (filas, caja, mesas, personal), 50-80 "
            "palabras.\n"
            '"temario": unidad y concepto del temario al que corresponde, con '
            'el formato "Unidad N — Concepto: justificación", 30-50 palabras.\n'
            '"pedagogia": qué aprende el estudiante con esta ficha y qué error '
            "conceptual previene, 40-70 palabras.\n"
            '"errores": entre 2 y 3 errores frecuentes al interpretar esta '
            "figura o tabla, separados por ' • ' en una sola línea, 40-70 "
            "palabras.\n"
            '"no_aplica": al menos 2 técnicas del temario que NO se aplican en '
            "esta figura o tabla y la razón concreta (tipo de variable, "
            "supuestos incumplidos, objetivo del análisis o costo "
            "computacional), 60-100 palabras.\n"
            "Cada valor debe ser una cadena de texto en español formal, sin "
            "saltos de línea internos salvo en 'errores', sin Markdown y sin "
            "emojis."
        )

    elif doble_capa:
        formato = (
            "Devuelve EXACTAMENTE dos párrafos separados por una línea que contenga "
            "únicamente tres barras verticales: |||\n"
            "Párrafo 1 (ANÁLISIS TÉCNICO): interpretación formal con las métricas y la "
            "gráfica (si se adjunta), en 80-110 palabras.\n"
            "Párrafo 2 (EN PALABRAS SIMPLES): la MISMA idea explicada como si se la "
            "contaras a alguien sin formación estadística (director de cafetería, docente "
            "de otra área). Cero jerga, cero siglas técnicas, usa analogías cotidianas y "
            "explica qué significa el resultado en la vida real, en 70-100 palabras."
        )
    else:
        formato = "Devuelve un solo párrafo técnico de 90-130 palabras."

    prompt = f"""
Eres un científico de datos senior asesorando un proyecto de Maestría en Inteligencia Artificial
sobre una cafetería universitaria (horario matutino 7:00 a 11:00 AM).

Sección del reporte: {seccion}
Métricas reales obtenidas desde la base de datos actual:
{json.dumps(datos_metricas, ensure_ascii=False, indent=2, default=str)}
{extra}
Instrucciones estrictas:
1. Interpreta los resultados y la gráfica (si se adjunta) usando ÚNICAMENTE los datos proporcionados.
2. {formato}
3. NO uses Markdown ni emojis.
4. Español formal, sin adornos publicitarios.
"""

    try:
        client = genai.Client(api_key=api_key)
        contents = [prompt]
        if imagen_png_bytes:
            contents.append(
                types.Part.from_bytes(data=imagen_png_bytes, mime_type="image/png")
            )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        if response and response.text and response.text.strip():
            if tipo == "didactico":
                result = _parse_campos_didacticos(response.text, fallback)
                _cache_set(key, json.dumps(result, ensure_ascii=False))
                with _CACHE_LOCK:
                    _CACHE_MISSES += 1
                return result
            raw = _limpiar_texto_pdf(response.text)
            if tipo == "decisiones":
                result = {"tecnico": raw, "simple": None}
            elif "|||" in raw:
                partes = [p.strip() for p in raw.split("|||") if p.strip()]
                result = {
                    "tecnico": partes[0],
                    "simple": partes[1] if len(partes) > 1 else None,
                }
            else:
                result = {"tecnico": raw, "simple": None}
            _cache_set(key, json.dumps(result, ensure_ascii=False))
            with _CACHE_LOCK:
                _CACHE_MISSES += 1
            return result
    except Exception:
        pass

    return fallback