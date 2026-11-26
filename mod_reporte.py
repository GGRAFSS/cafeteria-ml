"""
Módulo de Generación de Reporte PDF
Compila resultados de las 4 unidades + análisis IA crítico + plan operativo.

v3.3:
- Cada gráfica/tabla tiene: análisis IA (técnico + simple) + preguntas guía
  + decisiones operativas sugeridas por IA.
- Dendrograma ahora tiene bloque completo (análisis + preguntas + decisiones).
- Riesgos incluye decisiones de mitigación.
- 27 tareas IA en paralelo (16 análisis + 11 decisiones).
- Caché de respuestas IA + fallbacks robustos.

Fase 3B:
- 15 capítulos numerados con contenido real (sin esqueletos) + Resumen
  Ejecutivo como front matter sin numerar.
- Captions automáticos: 8 figuras + 7 tablas clave en sus índices.

Fase 5A (v5.0):
- Reporte Didáctico: método ficha_didactica() + generar_reporte_didactico().
- Cada figura/tabla tiene una ficha de 7 bloques, incluido el bloque
  "¿qué técnicas NO se aplican aquí y por qué?".

Fase 5B (v5.1):
- Contenido real: las 15 fichas se generan con analizar_seccion_ia(
  tipo="didactico") en paralelo y un respaldo curado por bloque.
"""
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

import data_manager as dm
import stats_utils as su
import mod_supervisado as ms
import mod_no_supervisado as mns
import mod_refuerzo as mr
from mod_ia_analista import CAMPOS_DIDACTICOS, analizar_seccion_ia


# ============================================================
# HELPERS
# ============================================================
def _fig_a_png(fig, dpi=100):
    if fig is None:
        return None
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def semaforo(metrica: str, valor: float) -> str:
    if valor is None:
        return "-"
    reglas = {
        "R2":         [(0.70, "VERDE"), (0.40, "AMARILLO"), (-1.0, "ROJO")],
        "F1":         [(0.70, "VERDE"), (0.50, "AMARILLO"), (-1.0, "ROJO")],
        "AUC":        [(0.80, "VERDE"), (0.65, "AMARILLO"), (-1.0, "ROJO")],
        "Accuracy":   [(0.80, "VERDE"), (0.60, "AMARILLO"), (-1.0, "ROJO")],
        "Silhouette": [(0.50, "VERDE"), (0.25, "AMARILLO"), (-1.0, "ROJO")],
    }
    for umbral, nivel in reglas.get(metrica, []):
        if valor >= umbral:
            return nivel
    return "-"


# ============================================================
# NIVEL 5 — nivel de muestra (N real de la base de datos)
# ============================================================
def _nivel_muestra(n):
    '''Devuelve (color, texto, permite_ensembles, permite_dl).

    Umbrales: n<100 ROJO; 100-199 AMARILLO; 200-499 VERDE; 500-999 VERDE+;
    n>=1000 VERDE++. Lo usan los textos de análisis IA (_preparar_tareas_ia)
    y el cuerpo del reporte (_construir_contenido).
    '''
    if n < 100:
        return ("ROJO", "INSUFICIENTE para generalizar", False, False)
    elif n < 200:
        return ("AMARILLO", "PRELIMINAR, ampliar recomendado", False, False)
    elif n < 500:
        return ("VERDE", "SUFICIENTE para conclusiones estables", False, False)
    elif n < 1000:
        return ("VERDE+", "EXCELENTE, permite ensembles", True, False)
    else:
        return ("VERDE++", "EXCELENTE, permite ensembles y Deep Learning tabular", True, True)


# ============================================================
# ASSETS (logos institucionales)
# ============================================================
def _get_asset_path(nombre: str) -> str | None:
    """Ruta absoluta a un archivo de assets/, o None si no existe.

    Usa Path(__file__) para resolverse igual en local y en Streamlit Cloud.
    """
    from pathlib import Path
    p = Path(__file__).parent / "assets" / nombre
    return str(p) if p.exists() else None


# ============================================================
# REPORTE DIDÁCTICO (v5.1)
# ============================================================
# Color de la etiqueta de cada bloque de la ficha didáctica.
_COLOR_CAMPO_DIDACTICO = {
    "que_se_ve": (0, 51, 102),      # azul institucional
    "tecnica": (0, 51, 102),
    "simple": (20, 90, 50),         # verde: lenguaje llano
    "temario": (90, 60, 130),       # violeta: temario
    "pedagogia": (150, 60, 20),     # naranja: pedagogía
    "errores": (170, 30, 30),       # rojo: errores frecuentes
    "no_aplica": (120, 70, 0),      # ámbar oscuro: bloque crítico
}

# Texto de seguridad: si un bloque llegara vacío se dibuja este aviso.
# No debería ocurrir: el respaldo curado de la Fase 5B trae los siete bloques.
_BLOQUE_SIN_TEXTO = "Contenido no disponible para este bloque."

# Qué responde cada bloque (se usa en la página introductoria del reporte).
_DESCRIPCION_CAMPO = {
    "que_se_ve": "Descripción visual objetiva: qué muestra la figura o tabla.",
    "tecnica": "Lectura técnica con los valores reales del proyecto.",
    "simple": "La misma idea sin jerga, con una analogía cotidiana.",
    "temario": "Unidad y concepto del temario que ilustra.",
    "pedagogia": "Qué habilidad entrena y qué error conceptual previene.",
    "errores": "Interpretaciones equivocadas frecuentes.",
    "no_aplica": "Técnicas del temario que NO sirven aquí y por qué.",
}


# ============================================================
# CLASE PDF
# ============================================================
class ReporteCafeteria(FPDF):
    # Valores por defecto a nivel de clase: fpdf2 puede invocar header()/
    # footer() antes de que _init_registros() se ejecute.
    indice_caps: dict = {}
    indice_figs: dict = {}
    indice_tabs: dict = {}
    fig_counter: int = 0
    tab_counter: int = 0
    cap_actual: str = ""
    total_pages: int = 0
    page_skip_header: bool = False
    page_skip_footer: bool = False
    _paginas_sin_header: set = set()
    _paginas_sin_footer: set = set()
    modo: str = "final"

    def __init__(self):
        super().__init__()
        self.FONT = "Helvetica"
        self._register_unicode_fonts()
        self._init_registros()
        # Flag interno para distinguir fases (borrador vs final)
        self.modo = "final"

    def _register_unicode_fonts(self):
        """Registra DejaVu Sans (Unicode) desde los TTF de matplotlib.

        fpdf2 >= 2.8 ya no acepta `uni=` en add_font(); la firma moderna
        detecta el mapeo Unicode automáticamente.
        """
        try:
            import matplotlib
            from pathlib import Path
            font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
            variants = {"": "DejaVuSans.ttf", "B": "DejaVuSans-Bold.ttf",
                        "I": "DejaVuSans-Oblique.ttf", "BI": "DejaVuSans-BoldOblique.ttf"}
            for style, fname in variants.items():
                fpath = font_dir / fname
                if not fpath.exists():
                    return
                self.add_font("DejaVu", style, str(fpath))
            self.FONT = "DejaVu"
        except Exception:
            pass

    # ---------- REGISTROS / ÍNDICES ----------
    def _init_registros(self):
        """Limpia los dicts de índices al inicio de cada generación."""
        self.indice_caps = {}      # {alias: pagina}
        self.indice_figs = {}      # {alias: pagina}
        self.indice_tabs = {}      # {alias: pagina}
        self.fig_counter = 0
        self.tab_counter = 0
        self.cap_actual = ""
        self.total_pages = 0
        self.page_skip_header = False
        self.page_skip_footer = False
        self._paginas_sin_header = set()
        self._paginas_sin_footer = set()

    def _saltar_header_footer(self, skip: bool):
        """Activa/desactiva encabezado y pie desde la página actual.

        Se guarda el número de página en un set porque el pie se dibuja al
        CERRAR la página (siguiente add_page() u output()), no al inicio.
        """
        self.page_skip_header = skip
        self.page_skip_footer = skip
        if skip:
            pagina = self.page_no() or 1     # portada: aún no hay página creada
            self._paginas_sin_header.add(pagina)
            self._paginas_sin_footer.add(pagina)

    def header(self):
        if self.page_skip_header or self.page_no() in self._paginas_sin_header:
            return
        if not self.cap_actual:
            # Portada / índices (front matter): sin encabezado. Cubre también
            # las páginas de desbordamiento de un índice.
            return
        self.set_font(self.FONT, "B", 8)
        self.set_text_color(120, 120, 120)
        # Título corto a la izquierda
        self.cell(95, 5, "Cafetería Universitaria — Aprendizaje Automático", 0, 0, "L")
        # Capítulo actual a la derecha
        self.cell(0, 5, self.cap_actual, 0, 1, "R")
        # Línea horizontal fina
        self.set_draw_color(200, 200, 200)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def footer(self):
        if self.page_skip_footer or self.page_no() in self._paginas_sin_footer:
            return
        self.set_y(-15)
        self.set_font(self.FONT, "I", 8)
        self.set_text_color(150, 150, 150)
        if self.total_pages > 0:
            texto = f"Página {self.page_no()} de {self.total_pages}"
        else:
            texto = f"Página {self.page_no()}"
        self.cell(0, 10, texto, 0, 0, "C")

    # ---------- BÁSICOS ----------
    def titulo_capitulo(self, num, titulo):
        """Abre página nueva, registra el capítulo en el índice y lo dibuja.

        Se usa `page_no() > 0` (v3.3 usaba `add_page()` incondicional y la
        propuesta `> 1` rompía el render borrador, donde la portada ES la
        página 1: el capítulo 1 se dibujaba encima de ella).

        `cap_actual` se actualiza ANTES de add_page() para que el encabezado
        de la primera página del capítulo ya muestre el capítulo correcto.
        """
        alias = f"{num}. {titulo}"
        self.cap_actual = f"Cap. {num}: {titulo[:40]}"
        if self.page_no() > 0:
            self.add_page()
        self.indice_caps[alias] = self.page_no()
        self.set_font(self.FONT, "B", 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, alias, 0, 1, "L")
        self.set_draw_color(0, 51, 102)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(5)

    def subtitulo(self, titulo):
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "B", 11)
        self.set_text_color(51, 51, 51)
        self.cell(0, 8, titulo, 0, 1, "L")
        self.ln(2)

    def parrafo(self, texto):
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "", 9)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, texto)
        self.ln(3)

    def nota_doble(self, tecnico: str, simple: str | None):
        self.set_x(self.l_margin)
        if tecnico:
            self.set_font(self.FONT, "I", 8)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 4, f">> Análisis técnico: {tecnico}")
            self.ln(1)
        if simple:
            self.set_x(self.l_margin)
            self.set_font(self.FONT, "", 9)
            self.set_text_color(20, 90, 50)
            self.multi_cell(0, 4.5, f"En palabras simples: {simple}")
            self.set_text_color(0, 0, 0)
        self.ln(3)

    def preguntas_guia(self, preguntas: list[str]):
        if not preguntas:
            return
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "I", 8)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 4, "Preguntas para profundizar:")
        for i, p in enumerate(preguntas, 1):
            self.set_x(self.l_margin)
            self.multi_cell(0, 4, f"  {i}. {p}")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    # ---------- DECISIONES SUGERIDAS ----------
    def decisiones_sugeridas(self, texto_o_lista):
        """Lista de decisiones operativas. Acepta string multilínea o lista."""
        if not texto_o_lista:
            return
        self.set_x(self.l_margin)

        if isinstance(texto_o_lista, str):
            lineas = [l.strip() for l in texto_o_lista.split("\n") if l.strip()]
        else:
            lineas = [str(l).strip() for l in texto_o_lista if str(l).strip()]

        # Filtrar viñetas vacías
        lineas_limpias = []
        for l in lineas:
            l_clean = l.lstrip("•-* ").strip()
            if l_clean:
                lineas_limpias.append(l_clean)
        if not lineas_limpias:
            return

        self.set_font(self.FONT, "B", 9)
        self.set_text_color(0, 70, 130)   # azul oscuro diferenciador
        self.multi_cell(0, 4.5, "Decisiones sugeridas:")
        self.set_font(self.FONT, "", 9)
        self.set_text_color(0, 0, 0)
        for l in lineas_limpias:
            self.set_x(self.l_margin + 2)
            self.multi_cell(0, 4.5, f"• {l}")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    # ---------- TABLAS ----------
    def tabla_metrica(self, filas, ancho1=60, ancho2=60,
                      encabezados=("Métrica", "Valor")):
        """Tabla de dos columnas con encabezado personalizable.

        `encabezados` permite rótulos distintos de "Métrica"/"Valor" (p. ej.
        "Concepto del temario"/"¿Aplica al proyecto?" en el cap. 3). El default
        conserva el comportamiento previo para todos los llamadores existentes.

        Fix de desbordes (v5.1): la columna de valores usa multi_cell() para
        envolver el texto dentro del recuadro; si el valor es largo se baja a
        8 pt y, si alguna palabra no cabe en la columna (p. ej. una firma de
        sklearn), se corta por caracteres (wrapmode="CHAR").
        """
        ancho_util = self.w - self.l_margin - self.r_margin
        if ancho1 + ancho2 > ancho_util:
            ancho2 = ancho_util - ancho1

        def _encabezado():
            self.set_font(self.FONT, "B", 9)
            self.set_fill_color(0, 51, 102)
            self.set_text_color(255, 255, 255)
            self.set_x(self.l_margin)
            self.cell(ancho1, 7, str(encabezados[0]), 1, 0, "C", True)
            self.cell(ancho2, 7, str(encabezados[1]), 1, 1, "C", True)
            self.set_text_color(0, 0, 0)

        def _wrap(texto, ancho):
            """CHAR si alguna palabra no cabe en el ancho; WORD si cabe."""
            limite = ancho - 2 * self.c_margin
            if any(self.get_string_width(t) > limite for t in texto.split()):
                return "CHAR"
            return "WORD"

        _encabezado()

        for k, v in filas:
            k_txt, v_txt = str(k), str(v)
            tam_v = 8 if len(v_txt) > 42 else 9
            alto_v = 4.6 if tam_v == 8 else 5.2

            # Evita partir una fila entre páginas (mínimo 2 renglones).
            if self.get_y() + 2 * alto_v + 2 > self.h - self.b_margin - 5:
                self.add_page()
                _encabezado()

            x_ini = self.l_margin
            y_ini = self.get_y()

            self.set_font(self.FONT, "", 8 if len(k_txt) > 42 else 9)
            self.set_xy(x_ini, y_ini)
            self.multi_cell(ancho1, 5, k_txt, 1, "L",
                            wrapmode=_wrap(k_txt, ancho1))
            y_col1 = self.get_y()

            self.set_font(self.FONT, "", tam_v)
            self.set_xy(x_ini + ancho1, y_ini)
            self.multi_cell(ancho2, alto_v, v_txt, 1, "L",
                            wrapmode=_wrap(v_txt, ancho2))
            y_col2 = self.get_y()

            self.set_xy(x_ini, max(y_col1, y_col2))
        self.ln(3)

    def tabla_larga(self, filas, ancho1=55, ancho2=125):
        self.set_x(self.l_margin)
        ancho_util = self.w - self.l_margin - self.r_margin
        if ancho1 + ancho2 > ancho_util:
            ancho2 = ancho_util - ancho1

        def _encabezado():
            self.set_font(self.FONT, "B", 9)
            self.set_fill_color(0, 51, 102)
            self.set_text_color(255, 255, 255)
            self.set_xy(self.l_margin, self.get_y())
            self.cell(ancho1, 7, "Término", 1, 0, "C", True)
            self.cell(ancho2, 7, "Definición", 1, 1, "C", True)
            self.set_font(self.FONT, "", 8)
            self.set_text_color(0, 0, 0)

        _encabezado()

        for k, v in filas:
            altura_min = 6
            if self.get_y() + altura_min > self.h - self.b_margin - 5:
                self.add_page()
                _encabezado()
            x_ini = self.l_margin
            y_ini = self.get_y()
            self.set_xy(x_ini, y_ini)
            self.multi_cell(ancho1, 5, str(k), 1, "L")
            y_tras_col1 = self.get_y()
            self.set_xy(x_ini + ancho1, y_ini)
            self.multi_cell(ancho2, 5, str(v), 1, "L")
            y_tras_col2 = self.get_y()
            self.set_xy(x_ini, max(y_tras_col1, y_tras_col2))
        self.ln(3)

    def agregar_figura(self, fig_o_bytes, titulo="", return_bytes=False):
        if fig_o_bytes is None:
            return None
        if isinstance(fig_o_bytes, (bytes, bytearray)):
            img_bytes = bytes(fig_o_bytes)
        else:
            img_bytes = _fig_a_png(fig_o_bytes)
            if img_bytes is None:
                return None
        self.set_x(self.l_margin)
        if titulo:
            self.set_font(self.FONT, "I", 8)
            self.set_text_color(80, 80, 80)
            self.cell(0, 5, titulo, 0, 1, "C")
        self.image(io.BytesIO(img_bytes), x=30, w=150)
        self.ln(3)
        return img_bytes if return_bytes else None

    # ---------- CAPTIONS (índices de figuras y tablas) ----------
    def caption_figura(self, alias: str):
        """Registra el caption de una figura y lo dibuja DEBAJO de ella."""
        self.fig_counter += 1
        texto = f"Figura {self.fig_counter}. {alias}"
        self.indice_figs[texto] = self.page_no()
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "I", 8)
        self.set_text_color(80, 80, 80)
        self.cell(0, 5, texto, 0, 1, "C")
        self.ln(2)

    def caption_tabla(self, alias: str):
        """Registra y dibuja el caption de una tabla (ARRIBA de la tabla)."""
        self.tab_counter += 1
        texto = f"Tabla {self.tab_counter}. {alias}"
        self.indice_tabs[texto] = self.page_no()
        self.set_x(self.l_margin)
        self.set_font(self.FONT, "B", 8)
        self.set_text_color(80, 80, 80)
        self.cell(0, 5, texto, 0, 1, "L")
        self.ln(1)

    # ---------- FICHAS DIDÁCTICAS (Reporte Didáctico, v5.1) ----------
    def ficha_didactica(self, numero: int, titulo: str, campos: dict,
                        referencia: str | None = None,
                        cita_fuente: str | None = None):
        """Dibuja una ficha didáctica de siete bloques para una figura o tabla.

        `campos` es el dict que devuelve
        `analizar_seccion_ia(..., tipo="didactico")`: las siete claves de
        CAMPOS_DIDACTICOS. Si falta una clave se dibuja un aviso neutral, de
        modo que la ficha nunca queda vacía.

        `referencia` indica de dónde sale el objeto explicado (p. ej.
        "Figura 3 del reporte completo") y `cita_fuente` la procedencia del
        texto (IA o respaldo automático).
        """
        ancho_util = self.w - self.l_margin - self.r_margin
        if self.get_y() > self.h - self.b_margin - 45:
            self.add_page()

        # --- Banda de título ---
        encabezado = f"  Ficha D{numero}. {titulo}"
        self.set_font(self.FONT, "B", 11)
        if self.get_string_width(encabezado) > ancho_util - 4:
            self.set_font(self.FONT, "B", 9)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        self.set_x(self.l_margin)
        self.cell(ancho_util, 9, encabezado, 0, 1, "L", True)
        self.set_text_color(0, 0, 0)

        # --- Procedencia del objeto explicado ---
        notas = [t for t in (referencia, cita_fuente) if t]
        if notas:
            self.set_font(self.FONT, "I", 7.5)
            self.set_text_color(90, 90, 90)
            for nota in notas:
                self.set_x(self.l_margin)
                self.multi_cell(ancho_util, 4, nota, 0, "L")
            self.set_text_color(0, 0, 0)
        self.ln(2)

        # --- Siete bloques de contenido ---
        for clave, etiqueta in CAMPOS_DIDACTICOS:
            texto = (campos or {}).get(clave) or _BLOQUE_SIN_TEXTO
            texto = str(texto).strip()

            # Salto de página preventivo antes de cada bloque
            if self.get_y() > self.h - self.b_margin - 28:
                self.add_page()
                self.set_x(self.l_margin)
                self.set_font(self.FONT, "I", 7.5)
                self.set_text_color(90, 90, 90)
                self.cell(0, 4, f"Ficha D{numero} (continuación)", 0, 1, "L")
                self.set_text_color(0, 0, 0)
                self.ln(1)

            color = _COLOR_CAMPO_DIDACTICO.get(clave, (0, 51, 102))
            self.set_x(self.l_margin)
            if clave == "no_aplica":
                # Bloque crítico: fondo ámbar para que no pase desapercibido
                self.set_fill_color(255, 243, 205)
                self.set_text_color(*color)
                self.set_font(self.FONT, "B", 9)
                self.cell(ancho_util, 6, f" {etiqueta}", 0, 1, "L", True)
                self.set_x(self.l_margin)
                self.set_font(self.FONT, "", 9)
                self.multi_cell(ancho_util, 4.6, texto, 0, "L")
            else:
                self.set_text_color(*color)
                self.set_font(self.FONT, "B", 9)
                self.multi_cell(ancho_util, 5, etiqueta, 0, "L")
                self.set_x(self.l_margin)
                self.set_text_color(0, 0, 0)
                self.set_font(self.FONT, "", 9)
                self.multi_cell(ancho_util, 4.6, texto, 0, "L")
            self.set_text_color(0, 0, 0)
            self.ln(2.5)

    # ---------- PORTADA INSTITUCIONAL ----------
    def portada_institucional(self):
        """Portada tipo tesis con los dos logos institucionales.

        No lleva encabezado ni pie: se marca la página 1 como excluida.
        """
        # Desactivar header/footer en esta página
        self._saltar_header_footer(True)
        self.add_page()

        # Logos (PNG con transparencia; fpdf2 calcula el alto por aspecto)
        logo_tec = _get_asset_path("logo_tecnm_virtual.png")
        logo_ia = _get_asset_path("logo_maestria_ia.png")

        if logo_tec:
            # Centrado: x = (210 - 26) / 2 = 92
            self.image(logo_tec, x=92, y=18, w=26)
        if logo_ia:
            # Centrado abajo: x = (210 - 28) / 2 = 91
            self.image(logo_ia, x=91, y=235, w=28)

        # Institución
        self.set_y(70)
        self.set_font(self.FONT, "B", 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 8, "INSTITUTO TECNOLÓGICO DE TIJUANA", 0, 1, "C")
        self.set_font(self.FONT, "", 11)
        self.cell(0, 6, "Maestría en Inteligencia Artificial", 0, 1, "C")
        self.ln(8)

        # Línea decorativa superior
        self.set_draw_color(0, 51, 102)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(10)

        # Título del proyecto
        self.set_font(self.FONT, "B", 18)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 9, "Reporte de Aprendizaje Automático\nCafetería Universitaria", 0, "C")
        self.ln(3)
        self.set_font(self.FONT, "I", 12)
        self.cell(0, 7, "Proyecto Integrador Final", 0, 1, "C")
        self.ln(6)

        # Línea decorativa inferior
        self.set_draw_color(0, 51, 102)
        self.line(60, self.get_y(), 150, self.get_y())
        self.ln(15)

        # Autores
        self.set_font(self.FONT, "", 11)
        self.set_text_color(0, 0, 0)
        self.cell(0, 6, "Autores:", 0, 1, "C")
        self.set_font(self.FONT, "B", 11)
        self.cell(0, 6, "Rodríguez Avilez Selene", 0, 1, "C")
        self.cell(0, 6, "Raygoza Flores Gabriel", 0, 1, "C")
        self.ln(8)

        # Docente
        self.set_font(self.FONT, "", 10)
        self.cell(0, 6, "Docente: Dra. Claudia Ibeth González Berrelleza", 0, 1, "C")
        self.cell(0, 6, "Instituto Tecnológico de Tijuana - TecNM", 0, 1, "C")

        # Fecha y lugar
        self.ln(10)
        self.set_font(self.FONT, "I", 10)
        self.cell(0, 6, "Tijuana, Baja California", 0, 1, "C")
        self.cell(0, 6, f"Noviembre {datetime.now().year}", 0, 1, "C")

        # URLs al pie
        self.set_y(270)
        self.set_font(self.FONT, "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 4, "App: https://cafeteria-ml-aprendizaje-auto.streamlit.app", 0, 1, "C")
        self.cell(0, 4, "Repo: https://github.com/GGRAFSS/cafeteria-ml", 0, 1, "C")

        # Reactivar header/footer para las páginas siguientes
        self._saltar_header_footer(False)

    # ---------- ÍNDICES ----------
    def _cabecera_indice(self, titulo: str):
        """Prepara una página de índice: sin encabezado, con pie."""
        self.page_skip_header = True      # el encabezado no va en esta página
        self.page_skip_footer = False     # el pie SÍ se dibuja
        self.add_page()
        self.page_skip_header = False
        self.cap_actual = ""              # el front matter no lleva cabecera
        self.set_font(self.FONT, "B", 16)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, titulo, 0, 1, "L")
        self.ln(5)
        self.set_font(self.FONT, "", 10)
        self.set_text_color(0, 0, 0)

    def _fila_indice(self, titulo: str, pagina: int):
        """Fila del índice: título + puntos + página. Trunca si el título
        excede el ancho disponible."""
        self.set_x(self.l_margin)

        self.set_font(self.FONT, "", 9)
        self.set_text_color(0, 0, 0)

        ancho_total = self.w - self.l_margin - self.r_margin  # ~190mm
        ancho_pagina = 15
        ancho_max_texto = ancho_total - ancho_pagina - 5

        # Ancho real del título
        ancho_texto = self.get_string_width(titulo) + 2
        titulo_mostrar = titulo

        # Truncar si excede el espacio
        if ancho_texto > ancho_max_texto:
            while ancho_texto > ancho_max_texto and len(titulo_mostrar) > 8:
                titulo_mostrar = titulo_mostrar[:-1]
                ancho_texto = self.get_string_width(titulo_mostrar + "…") + 2
            titulo_mostrar = titulo_mostrar + "…"
            ancho_texto = self.get_string_width(titulo_mostrar) + 2

        # Título
        self.cell(ancho_texto, 6, titulo_mostrar, 0, 0, "L")

        # Puntos
        ancho_puntos = ancho_total - ancho_texto - ancho_pagina
        if ancho_puntos >= 2:
            self.set_font(self.FONT, "", 8)
            self.set_text_color(180, 180, 180)
            ancho_punto = self.get_string_width(".")
            num_puntos = max(int(ancho_puntos / ancho_punto) - 1, 3)
            self.cell(ancho_puntos, 6, "." * num_puntos, 0, 0, "L")
            self.set_text_color(0, 0, 0)

        # Página
        self.set_font(self.FONT, "B", 9)
        self.cell(ancho_pagina, 6, str(pagina), 0, 1, "R")

        # Restaurar tipografía base para quien dibuje después
        self.set_font(self.FONT, "", 10)

    def indice_contenido(self):
        """Dibuja el índice de contenido usando self.indice_caps."""
        self._cabecera_indice("Índice de Contenido")
        for titulo, pagina in self.indice_caps.items():
            self._fila_indice(titulo, pagina)
        self._saltar_header_footer(False)

    def indice_figuras(self):
        """Dibuja el índice de figuras usando self.indice_figs."""
        self._cabecera_indice("Índice de Figuras")
        for titulo, pagina in self.indice_figs.items():
            self._fila_indice(titulo, pagina)
        self._saltar_header_footer(False)

    def indice_tablas(self):
        """Dibuja el índice de tablas usando self.indice_tabs."""
        self._cabecera_indice("Índice de Tablas")
        for titulo, pagina in self.indice_tabs.items():
            self._fila_indice(titulo, pagina)
        self._saltar_header_footer(False)


# ============================================================
# FASE 1: CONTEXTO
# ============================================================
def _construir_contexto(df: pd.DataFrame) -> dict:
    df_limpio = dm.clean_data(df)
    resumen_exec = su.executive_summary(df, df_limpio)
    desc = df[["Hora_Llegada", "Tiempo_Espera_Min",
               "Tiempo_Servicio_Min", "Consumo_Total_MXN"]].describe()
    outliers_summary, _ = su.detect_outliers_iqr(df)

    res_reg = ms.entrenar_regresion(df)
    mejores = {}
    for modelo in ["Regresión Logística", "KNN", "SVM", "Árbol de Decisión"]:
        r = ms.entrenar_clasificacion(df, modelo)
        if not r.get("error"):
            mejores[modelo] = r

    res_km = mns.ejecutar_kmeans(df, n_clusters=3)
    res_jer = mns.ejecutar_jerarquico(df)

    Q, historial = mr.ejecutar_q_learning(episodios=500, alpha=0.1, gamma=0.9)
    fig_ap, fig_q = mr.graficar_resultados(Q, historial)
    politica = mr.obtener_politica_optima(Q)
    politica_records = politica.to_dict(orient="records")

    contexto = {
        "total_registros_bd": len(df),
        "registros_limpios": len(df_limpio),
        "outliers_detectados": resumen_exec["outliers"],
        "consumo_promedio_mxn": round(resumen_exec["consumo_prom"], 2),
        "consumo_total_mxn": round(resumen_exec["consumo_total"], 2),
        "espera_promedio_min": round(resumen_exec["espera_prom"], 2),
        "espera_maxima_min": round(resumen_exec["espera_max"], 2),
        "servicio_promedio_min": round(resumen_exec["servicio_prom"], 2),
        "porcentaje_hora_pico": round(resumen_exec["pct_pico"], 1),
        "satisfaccion_promedio": round(resumen_exec["sat_prom"], 2),
    }

    if not res_reg.get("error"):
        contexto["regresion"] = {
            "lr_mse": round(float(res_reg["metricas_lr"]["MSE"]), 2),
            "lr_r2": round(float(res_reg["metricas_lr"]["R2"]), 4),
            "dt_mse": round(float(res_reg["metricas_dt"]["MSE"]), 2),
            "dt_r2": round(float(res_reg["metricas_dt"]["R2"]), 4),
        }

    if mejores:
        resumen_modelos = {}
        for m, r in mejores.items():
            auc = r["metricas"].get("AUC")
            resumen_modelos[m] = {
                "Accuracy": round(float(r["metricas"]["Accuracy"]), 3),
                "Precision": round(float(r["metricas"]["Precision"]), 3),
                "Recall": round(float(r["metricas"]["Recall"]), 3),
                "F1": round(float(r["metricas"]["F1"]), 3),
                "AUC": round(float(auc), 3) if auc is not None else None,
            }
        mejor_nombre = max(mejores, key=lambda k: mejores[k]["metricas"]["F1"])
        contexto["clasificacion"] = {
            "mejor_modelo": mejor_nombre,
            "mejor_f1": round(float(mejores[mejor_nombre]["metricas"]["F1"]), 3),
            "mejor_auc": round(float(mejores[mejor_nombre]["metricas"].get("AUC") or 0.0), 3),
            "modelos": resumen_modelos,
        }

    if not res_km.get("error"):
        contexto["clustering_kmeans"] = {
            "k": 3,
            "silhouette": round(float(res_km["silhouette"]), 4),
        }

    contexto["refuerzo_q_learning"] = {
        "episodios": 500,
        "recompensa_prom_primeros_50": round(float(np.mean(historial[:50])), 2),
        "recompensa_prom_ultimos_50": round(float(np.mean(historial[-50:])), 2),
        "politica_optima": politica_records,
    }
    contexto["outliers_por_variable"] = (
        outliers_summary.to_dict(orient="records")
        if not outliers_summary.empty else []
    )

    return {
        "df_limpio": df_limpio,
        "resumen_exec": resumen_exec,
        "desc": desc,
        "outliers_summary": outliers_summary,
        "res_reg": res_reg,
        "mejores": mejores,
        "res_km": res_km,
        "res_jer": res_jer,
        "fig_ap": fig_ap,
        "fig_q": fig_q,
        "politica": politica,
        "politica_records": politica_records,
        "contexto": contexto,
    }


# ============================================================
# FASE 2: IA EN PARALELO (27 tareas)
# ============================================================
def _ejecutar_ia_paralelo(tareas_ia: list, fallbacks: dict) -> dict:
    resultados = {
        nombre: {"tecnico": fallbacks.get(nombre, ""), "simple": None}
        for nombre, _, _ in tareas_ia
    }
    # 4 workers: balance entre velocidad y rate limits de Gemini Flash
    with ThreadPoolExecutor(max_workers=4) as executor:
        futuros = {
            executor.submit(analizar_seccion_ia, *args, **kwargs): nombre
            for nombre, args, kwargs in tareas_ia
        }
        for fut in as_completed(futuros):
            nombre = futuros[fut]
            try:
                r = fut.result()
                if isinstance(r, dict) and r.get("tecnico"):
                    resultados[nombre] = r
            except Exception:
                pass
    return resultados


def _preparar_tareas_ia(ctx: dict, png: dict) -> tuple[list, dict]:
    contexto = ctx["contexto"]
    resumen_exec = ctx["resumen_exec"]
    len_df = contexto["total_registros_bd"]
    _color_df, _nivel_df, _permite_ens_df, _permite_dl_df = _nivel_muestra(len_df)

    calidad_datos = {
        "total_registros": len_df,
        "registros_validos": contexto["registros_limpios"],
        "porcentaje_outliers": round(contexto["outliers_detectados"] / max(len_df, 1) * 100, 2),
        "outliers_por_variable": contexto["outliers_por_variable"],
        "porcentaje_hora_pico": contexto["porcentaje_hora_pico"],
    }

    contexto_plan = {
        "resumen_general": contexto,
        "porcentaje_hora_pico": contexto["porcentaje_hora_pico"],
        "satisfaccion_promedio": contexto["satisfaccion_promedio"],
        "politica_baristas": ctx["politica_records"],
        "mejor_modelo_supervisado": contexto.get("clasificacion", {}).get("mejor_modelo", "N/A"),
    }

    # ---------- FALLBACKS DE ANÁLISIS ----------
    fallbacks = {
        "resumen": (
            f"El proyecto analiza {len_df} registros con consumo promedio de "
            f"${resumen_exec['consumo_prom']:.2f} MXN y satisfacción de "
            f"{resumen_exec['sat_prom']:.2f}/5. Mejor clasificador: "
            f"{contexto.get('clasificacion', {}).get('mejor_modelo', 'N/A')}."
        ),
        "calidad": (
            f"Se detectaron {resumen_exec['outliers']} outliers "
            f"({calidad_datos['porcentaje_outliers']:.1f}% del total). "
            f"El {contexto['porcentaje_hora_pico']:.1f}% corresponde a hora pico."
        ),
        "reg": (
            f"R² Lineal={contexto.get('regresion', {}).get('lr_r2', 'N/A')}; "
            f"R² Árbol={contexto.get('regresion', {}).get('dt_r2', 'N/A')}."
        ),
        "clf": (
            f"Mejor modelo: {contexto.get('clasificacion', {}).get('mejor_modelo', 'N/A')} "
            f"con F1={contexto.get('clasificacion', {}).get('mejor_f1', 'N/A')}."
        ),
        "km": (
            f"Silhouette k=3: {contexto.get('clustering_kmeans', {}).get('silhouette', 'N/A')}."
        ),
        "ana_dendro": (
            "El dendrograma agrupa clientes por similitud. Las ramas más largas indican "
            "separación clara entre grupos; los cortes óptimos aparecen donde la distancia "
            "salta con mayor pendiente."
        ),
        "rl": (
            f"Recompensa: {contexto['refuerzo_q_learning']['recompensa_prom_primeros_50']:.1f} "
            f"→ {contexto['refuerzo_q_learning']['recompensa_prom_ultimos_50']:.1f}."
        ),
        "plan": (
            "Ajustar turnos en hora pico, agilizar caja con pre-pago por app, "
            "y segmentar promociones por tipo de cliente."
        ),
        "riesgos": (
            "Riesgos: muestra acotada, Q-Learning sobre simulación, posible "
            "desbalance de clases. Próximos pasos: más datos reales y telemetría."
        ),
        "conclusiones": (
            f"El análisis sobre {len_df} registros respalda decisiones operativas "
            f"en horario matutino combinando modelos supervisados, no supervisados "
            f"y de refuerzo."
        ),
        # ---------- FASE 3B: capítulos 2, 3, 5, 6, 7 y 11 ----------
        "ana_objetivos": (
            "Los objetivos están alineados al ciclo de vida de ML: preparación, "
            "selección, entrenamiento, evaluación e implementación. Cada uno "
            "cubre una unidad del temario y se materializa en la app funcional."
        ),
        "ana_marco": (
            "El proyecto aplica correctamente supervisado, no supervisado y "
            "refuerzo. La exclusión de Deep Learning y GANs está justificada "
            f"por el tamaño y estructura del dataset ({len_df} registros "
            f"tabulares, nivel {_color_df}: {_nivel_df.lower()})."
        ),
        "ana_preprocesamiento": (
            "El pipeline aplica limpieza, normalización y codificación. "
            "StandardScaler es crítico para KNN y SVM; LabelEncoder es "
            "suficiente para las 3 categorías por variable."
        ),
        "ana_division": (
            "La división 70/30 con random_state=42 es reproducible y reserva "
            f"~30% de los datos para una evaluación honesta. Con {len_df} "
            "registros "
            + ("no conviene un tercer conjunto: la validación cruzada interna "
               "del SVM (cv=3) cumple esa función sin restar datos."
               if len_df < 500 else
               "un tercer conjunto sería viable, pero se conserva 70/30 con "
               "validación cruzada interna (cv=3) para mantener comparabilidad "
               "con el resto del proyecto.")
        ),
        "ana_seleccion": (
            "Se comparan 2 regresores, 4 clasificadores, 3 algoritmos de "
            "agrupamiento y 1 de refuerzo: cobertura amplia del temario con "
            f"modelos clásicos adecuados a {len_df} registros tabulares "
            f"(nivel {_color_df}). Los ensembles y el Deep Learning quedan "
            "fuera por decisión metodológica del proyecto"
            + (": con este N tampoco serían viables." if not _permite_ens_df
               else ": con este N serían técnicamente viables, pero se "
                    "prioriza la interpretabilidad de los modelos clásicos.")
        ),
        "ana_validacion": (
            "Se observa overfitting en el Árbol (delta F1>0.15) y underfitting "
            "leve en Regresión Lineal (R² bajo en train). El mejor modelo para "
            "desplegar es el que minimiza delta manteniendo F1 alto."
        ),
        "dec_preprocesamiento": (
            "• Documentar el proceso de captura para evitar outliers por error.\n"
            "• Revisar el 5% de registros con Consumo>200 MXN.\n"
            "• Ampliar la muestra antes de aplicar técnicas avanzadas."
        ),
        "dec_seleccion": (
            "• Mantener el modelo más simple que alcance F1 ≥ 0.70.\n"
            "• Documentar la justificación de cada exclusión metodológica.\n"
            "• Reintroducir Random Forest cuando n > 500 registros.\n"
            "• Revisar LDA/QDA si se necesita reducir dimensionalidad."
        ),
        "dec_validacion": (
            "• Desplegar el modelo con menor delta train/test (no el mayor F1).\n"
            "• Ampliar muestra a ≥200 registros antes de generalizar.\n"
            "• Reentrenar trimestralmente con datos nuevos."
        ),
        # Decisiones
        "dec_resumen": (
            "• Aumentar personal de caja en 8:00-9:00 AM (mayor afluencia).\n"
            "• Lanzar promoción prepago por app para agilizar cobro.\n"
            "• Segmentar ofertas según los 3 clústeres de cliente detectados."
        ),
        "dec_calidad": (
            "• Documentar el proceso de captura para evitar outliers por error humano.\n"
            "• Revisar los registros con espera mayor a 25 minutos.\n"
            "• Ampliar la muestra a más de 200 registros antes de generalizar."
        ),
        "dec_reg": (
            "• Usar Regresión Lineal como modelo base (interpretable y suficiente).\n"
            "• Descartar el Árbol si no mejora el R² en más de 0.10.\n"
            "• Agregar variables como clima o promociones al formulario."
        ),
        "dec_clf": (
            "• Priorizar Recall sobre Precision (no perder clientes insatisfechos).\n"
            "• Mantener Regresión Logística como modelo de producción.\n"
            "• Recolectar más encuestas de satisfacción 1-3 (clase minoritaria).\n"
            "• Revisar mesas con espera mayor a 8 minutos (mayor predictor)."
        ),
        "dec_km": (
            "• Nombrar los 3 clústeres según perfil real (ej. estudiante exprés).\n"
            "• Diseñar una promoción específica por cada clúster.\n"
            "• Probar k=4 antes de fijar el modelo final de segmentación."
        ),
        "dec_dendro": (
            "• Cortar el dendrograma donde aparezcan 3 ramas claramente separadas.\n"
            "• Revisar clientes en ramas largas como casos atípicos a analizar.\n"
            "• Contrastar el resultado con K-Means antes de decidir el corte final."
        ),
        "dec_rl": (
            "• Aplicar política: 1 barista con afluencia baja, 2 con media, 3 con alta.\n"
            "• Validar la política con telemetría real durante una semana.\n"
            "• Ajustar recompensas si se prioriza satisfacción sobre ganancia."
        ),
        "dec_riesgos": (
            "• Recolectar datos en tiempo real para sustituir la simulación de Q-Learning.\n"
            "• Balancear clases de satisfacción antes de reentrenar el clasificador.\n"
            "• Programar revisión trimestral del modelo con datos nuevos."
        ),
    }

    tareas = [
        # ---------- ANÁLISIS ----------
        ("resumen", ("Resumen Ejecutivo del Proyecto", contexto, fallbacks["resumen"]), {
            "instruccion_extra": (
                "Genera un resumen ejecutivo (máx 140 palabras) que cubra: "
                "volumen de datos, consumo promedio, satisfacción promedio, "
                "mejor modelo supervisado con su F1, y la política óptima de "
                "baristas. Enfócate en decisiones ejecutivas, no en tecnicismos."
            ),
        }),
        ("calidad", ("Diagnóstico de Calidad de Datos y Outliers", calidad_datos, fallbacks["calidad"]), {
            "instruccion_extra": (
                "Identifica si el porcentaje de outliers o la proporción de "
                "hora pico sesga el entrenamiento. Menciona riesgos por "
                "registros incompletos y recomienda imputar, eliminar o mantener."
            ),
        }),
        ("reg", (
            "Regresión: lectura de gráfica + comparativa crítica",
            contexto.get("regresion", {}), fallbacks["reg"],
        ), {
            "imagen_png_bytes": png["reg"],
            "instruccion_extra": (
                "Interpreta la gráfica real vs predicho Y compara los dos "
                "modelos. NO repitas la tabla. Enfócate en: (1) patrón de "
                "dispersión (sesgo, outliers), (2) trade-off interpretabilidad "
                "(lineal) vs no linealidad (árbol), (3) cuándo conviene cada uno."
            ),
        }),
        ("clf", (
            "Clasificación: lectura de matriz + comparativa crítica",
            contexto.get("clasificacion", {}), fallbacks["clf"],
        ), {
            "imagen_png_bytes": png["cm"],
            "instruccion_extra": (
                "Interpreta la matriz de confusión Y compara los 4 algoritmos. "
                "NO repitas la tabla. Enfócate en: (1) trade-off Precision vs "
                "Recall (cuándo priorizar Recall para no perder insatisfechos), "
                "(2) por qué SVM/KNN/Árbol pueden ganar o perder vs Logística."
            ),
        }),
        ("km", (
            "Segmentación No Supervisada (K-Means y PCA)",
            contexto.get("clustering_kmeans", {}), fallbacks["km"],
        ), {
            "imagen_png_bytes": png["km_pca"],
            "instruccion_extra": (
                "Interpreta los clústeres en el plano PCA. Menciona qué tipo de "
                "cliente podría representar cada grupo y utilidad operativa."
            ),
        }),
        ("ana_dendro", (
            "Clustering Jerárquico: lectura del dendrograma",
            {"n_registros": len_df, "linkage": "ward"},
            fallbacks["ana_dendro"],
        ), {
            "imagen_png_bytes": png["jer"],
            "instruccion_extra": (
                "Interpreta la estructura del dendrograma. Identifica el número "
                "óptimo de clústeres mirando dónde las ramas se separan más. "
                "Menciona si hay outliers (ramas muy largas)."
            ),
        }),
        ("rl", (
            "Aprendizaje por Refuerzo (Q-Learning)",
            contexto["refuerzo_q_learning"], fallbacks["rl"],
        ), {
            "imagen_png_bytes": png["ap"],
            "instruccion_extra": (
                "Interpreta la curva de aprendizaje y la política óptima. "
                "Indica si el agente converge y qué decisión operativa sugiere "
                "para afluencia Baja/Media/Alta."
            ),
        }),
        ("plan", ("Plan de Acción Operativo Priorizado", contexto_plan, fallbacks["plan"]), {
            "instruccion_extra": (
                "Genera 4-5 acciones concretas y priorizadas para la cafetería "
                "en horario 7-11 AM. Cada acción con impacto esperado y "
                "esfuerzo. Incluye turnos, agilización de caja, promociones "
                "segmentadas y gestión de hora pico."
            ),
        }),
        ("riesgos", ("Riesgos, Supuestos y Próximos Pasos", contexto, fallbacks["riesgos"]), {
            "instruccion_extra": (
                "Identifica 3-4 supuestos que podrían fallar (simulación vs "
                "operación real en Q-Learning, clases desbalanceadas, tamaño "
                "de muestra, estacionalidad). Propón 2-3 experimentos concretos."
            ),
        }),
        ("conclusiones", ("Conclusiones Generales del Proyecto", contexto, fallbacks["conclusiones"]), {}),
        # ---------- FASE 3B: ANÁLISIS DE LOS CAPÍTULOS NUEVOS ----------
        ("ana_objetivos", (
            "Objetivos del Proyecto",
            {
                "objetivo_general": (
                    "Desarrollar un sistema funcional de Aprendizaje Automático que "
                    "prediga el consumo, clasifique la satisfacción, segmente clientes "
                    "y optimice la asignación de personal en la cafetería universitaria "
                    "durante el horario matutino (7:00-11:00 AM)."
                ),
                "objetivos_especificos": [
                    "Preparación de datos",
                    "Selección de modelos",
                    "Entrenamiento y optimización",
                    "Evaluación del desempeño",
                    "Implementación funcional",
                ],
            },
            fallbacks["ana_objetivos"],
        ), {
            "instruccion_extra": (
                "Analiza la alineación de los objetivos con el ciclo de vida "
                "de ML. Comenta si cubren las 4 unidades del temario."
            ),
        }),
        ("ana_marco", (
            "Marco Conceptual",
            {
                "conceptos_aplicados": [
                    "Aprendizaje supervisado (regresión y clasificación)",
                    "Aprendizaje no supervisado (K-Means, Jerárquico, DBSCAN)",
                    "Aprendizaje por refuerzo (Q-Learning tabular)",
                ],
                "conceptos_excluidos": ["Deep Learning", "Transfer Learning", "GANs"],
                "n_registros": len_df,
            },
            fallbacks["ana_marco"],
        ), {
            "instruccion_extra": (
                "Justifica la exclusión de Deep Learning, Transfer Learning y "
                "GANs dado el tamaño y la naturaleza tabular del dataset."
            ),
        }),
        ("ana_preprocesamiento", (
            "Preprocesamiento de Datos",
            {
                "pipeline": [
                    "dropna + filtro Consumo<=250",
                    "StandardScaler sobre X_train",
                    "LabelEncoder en categóricas",
                    "PCA(2) solo para visualización",
                    "Outliers por IQR",
                ],
                "tecnicas_no_aplicadas": [
                    "One-Hot Encoding",
                    "Imputación por media/mediana",
                    "t-SNE / UMAP",
                ],
            },
            fallbacks["ana_preprocesamiento"],
        ), {
            "instruccion_extra": (
                "Explica el impacto del preprocesamiento en KNN y SVM y por qué "
                "LabelEncoder basta con 3 categorías por variable."
            ),
        }),
        ("ana_division", (
            "División de Datos (70/30)",
            {
                "n_registros": len_df,
                "test_size": 0.3,
                "random_state": 42,
                "validacion_cruzada": "CalibratedClassifierCV(cv=3) en SVM",
            },
            fallbacks["ana_division"],
        ), {
            "instruccion_extra": (
                "Evalúa si 70/30 es adecuado con el tamaño de muestra actual y "
                "explica por qué no se usa un conjunto de validación separado."
            ),
        }),
        ("ana_seleccion", (
            "Selección de Modelos",
            {
                "modelos_aplicados": {
                    "regresion": ["Lineal", "Árbol de Regresión"],
                    "clasificacion": ["Logística", "KNN", "SVM", "Árbol"],
                    "clustering": ["K-Means", "Jerárquico", "DBSCAN"],
                    "refuerzo": ["Q-Learning tabular"],
                },
                "modelos_excluidos": [
                    "Regresión polinómica", "Naive Bayes",
                    "Random Forest / XGBoost", "GMM", "Spectral Clustering",
                    "DQN / Policy Gradient / A2C",
                ],
                "n_registros": len_df,
            },
            fallbacks["ana_seleccion"],
        ), {
            "instruccion_extra": (
                "Analiza la regla de selección aplicada (complejidad proporcional "
                "al tamaño del dataset) y si las exclusiones están justificadas."
            ),
        }),
        ("ana_validacion", (
            "Validación y Overfitting",
            {
                "n_registros": len_df,
                "delta_umbral": 0.15,
                "modelos_comparados": [
                    "Regresión Lineal (train vs test R²)",
                    "Árbol de Regresión (train vs test R²)",
                    *[f"{m} (train vs test F1)" for m in ctx["mejores"]],
                ],
                "mitigaciones": [
                    "max_depth=5 en árboles",
                    "Regularización L2 en Logística",
                    "CalibratedClassifierCV(cv=3) en SVM",
                    "División 70/30 sin fuga de test",
                ],
            },
            fallbacks["ana_validacion"],
        ), {
            "instruccion_extra": (
                "Interpreta los deltas train/test: distingue overfitting de "
                "underfitting y recomienda qué modelo desplegar."
            ),
        }),
        ("dec_preprocesamiento", ("Decisiones - Preprocesamiento", contexto, fallbacks["dec_preprocesamiento"]), {
            "tipo": "decisiones",
            "doble_capa": False,
            "instruccion_extra": "Sugiere 3-4 decisiones sobre manejo de datos.",
        }),
        ("dec_seleccion", ("Decisiones - Selección de Modelos", contexto, fallbacks["dec_seleccion"]), {
            "tipo": "decisiones",
            "doble_capa": False,
            "instruccion_extra": (
                "Sugiere 3-4 decisiones sobre qué modelos mantener y cuándo "
                "reintroducir los excluidos."
            ),
        }),
        ("dec_validacion", ("Decisiones - Validación", contexto, fallbacks["dec_validacion"]), {
            "tipo": "decisiones",
            "doble_capa": False,
            "instruccion_extra": "Sugiere 3-4 acciones de mitigación.",
        }),
        # ---------- DECISIONES ----------
        ("dec_resumen", ("Decisiones - Resumen Ejecutivo", contexto, fallbacks["dec_resumen"]), {
            "tipo": "decisiones",
            "doble_capa": False,
            "instruccion_extra": (
                "Basándote en los KPIs, sugiere 3-4 decisiones operativas "
                "accionables esta semana para la cafetería."
            ),
        }),
        ("dec_calidad", ("Decisiones - Calidad de Datos", calidad_datos, fallbacks["dec_calidad"]), {
            "tipo": "decisiones",
            "doble_capa": False,
            "instruccion_extra": (
                "Sugiere 3-4 decisiones para mejorar la calidad de los datos "
                "capturados y la robustez del entrenamiento."
            ),
        }),
        ("dec_reg", ("Decisiones - Regresión", contexto.get("regresion", {}), fallbacks["dec_reg"]), {
            "tipo": "decisiones",
            "doble_capa": False,
            "instruccion_extra": (
                "Sugiere 3-4 decisiones operativas a partir de las métricas de "
                "regresión (R², MSE). No repitas números, traduce a acciones."
            ),
        }),
        ("dec_clf", ("Decisiones - Clasificación", contexto.get("clasificacion", {}), fallbacks["dec_clf"]), {
            "tipo": "decisiones",
            "doble_capa": False,
            "instruccion_extra": (
                "Sugiere 3-4 decisiones operativas a partir de las métricas de "
                "clasificación. Prioriza acciones sobre personal, datos y modelo."
            ),
        }),
        ("dec_km", ("Decisiones - K-Means", contexto.get("clustering_kmeans", {}), fallbacks["dec_km"]), {
            "tipo": "decisiones",
            "doble_capa": False,
            "instruccion_extra": (
                "Sugiere 3-4 decisiones a partir de la segmentación K-Means."
            ),
        }),
        ("dec_dendro", ("Decisiones - Jerárquico", {"n_registros": len_df}, fallbacks["dec_dendro"]), {
            "tipo": "decisiones",
            "doble_capa": False,
            "instruccion_extra": (
                "Sugiere 3-4 decisiones operativas a partir del clustering "
                "jerárquico y su lectura del dendrograma."
            ),
        }),
        ("dec_rl", ("Decisiones - Refuerzo", contexto["refuerzo_q_learning"], fallbacks["dec_rl"]), {
            "tipo": "decisiones",
            "doble_capa": False,
            "instruccion_extra": (
                "Sugiere 3-4 decisiones operativas basadas en la política "
                "óptima de Q-Learning."
            ),
        }),
        ("dec_riesgos", ("Decisiones - Mitigación de Riesgos", contexto, fallbacks["dec_riesgos"]), {
            "tipo": "decisiones",
            "doble_capa": False,
            "instruccion_extra": (
                "Sugiere 3-4 acciones concretas para mitigar los riesgos "
                "identificados del proyecto."
            ),
        }),
    ]
    return tareas, fallbacks


# ============================================================
# REPORTE COMPLETO
# ============================================================
def generar_reporte_pdf(df: pd.DataFrame) -> tuple:
    if df.empty or len(df) < 20:
        return None, "Se necesitan al menos 20 registros para generar el reporte."

    try:
        # ---- FASE 1: contexto (se ejecuta UNA sola vez) ----
        ctx = _construir_contexto(df)
        res_reg = ctx["res_reg"]
        mejores = ctx["mejores"]
        res_km = ctx["res_km"]
        res_jer = ctx["res_jer"]
        contexto = ctx["contexto"]

        # ---- FASE 1.5: figuras a PNG ----
        png = {
            "reg": _fig_a_png(res_reg["fig"]) if not res_reg.get("error") else None,
            "cm": None, "roc": None,
            "km_codo": _fig_a_png(res_km["fig_codo"]) if not res_km.get("error") else None,
            "km_pca": _fig_a_png(res_km["fig_pca"]) if not res_km.get("error") else None,
            "jer": _fig_a_png(res_jer["fig_dendro"]) if not res_jer.get("error") else None,
            "ap": _fig_a_png(ctx["fig_ap"]),
            "q": _fig_a_png(ctx["fig_q"]),
        }
        if mejores:
            mejor_nombre_tmp = contexto["clasificacion"]["mejor_modelo"]
            png["cm"] = _fig_a_png(mejores[mejor_nombre_tmp]["fig_cm"])
            png["roc"] = _fig_a_png(mejores[mejor_nombre_tmp].get("fig_roc"))

        # ---- FASE 2 ----
        tareas, fallbacks = _preparar_tareas_ia(ctx, png)
        textos = _ejecutar_ia_paralelo(tareas, fallbacks)

        # ---- FASE 3: PDF (doble pasada) ----
        pdf, _info = _render_reporte(ctx, textos, png, df)

        return bytes(pdf.output()), None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error al generar reporte: {e}"


# ============================================================
# FASE 3: RENDER (DOBLE PASADA: BORRADOR + FINAL)
# ============================================================
def _render_reporte(ctx: dict, textos: dict, png: dict, df=None) -> tuple:
    """Render #1 (borrador) + medición de índices + Render #2 (final).

    Render #1 dibuja portada + resumen ejecutivo + capítulos y registra
    alias→página.
    La medición dibuja portada + los 3 índices en un PDF desechable para
    saber CUÁNTAS páginas ocupan realmente (no se asume un offset fijo:
    con 15 capítulos el índice de figuras puede ocupar más de una página).
    Render #2 dibuja portada + índices (con las páginas del borrador
    desplazadas por ese offset) + resumen ejecutivo + los capítulos.

    El Resumen Ejecutivo es front matter: va SIN número después de los
    índices, por eso se dibuja aparte y se registra a mano en el índice.

    `paginas_total` (= borrador + offset) se fija en el Render #2 antes de
    dibujar, de modo que el pie imprima "Página N de T" sin una 3ra pasada.

    Retorna (pdf_final, info). Lo pesado (modelos, PNG, Gemini) ya está
    calculado en ctx/textos/png y NO se vuelve a ejecutar.
    """
    # ---------- RENDER #1: BORRADOR ----------
    pdf1 = ReporteCafeteria()
    pdf1.modo = "borrador"
    pdf1.set_auto_page_break(auto=True, margin=15)
    pdf1.portada_institucional()
    _resumen_ejecutivo(pdf1, ctx, textos, png, df)
    _construir_contenido(pdf1, ctx, textos, png, df)
    paginas_borrador = pdf1.page_no()

    # ---------- MEDICIÓN: páginas que ocupan los índices ----------
    pdf_med = ReporteCafeteria()
    pdf_med.set_auto_page_break(auto=True, margin=15)
    pdf_med.portada_institucional()
    # Páginas FICTICIAS de 2 dígitos: el índice final mostrará números de 2
    # dígitos (offset=3 => 5..N por ahora) y así el ancho de cada renglón es
    # el mismo que en el PDF real, sin alterar el conteo de líneas por página.
    pdf_med.indice_caps = {k: 10 + i for i, k in enumerate(pdf1.indice_caps)}
    pdf_med.indice_figs = {k: 10 + i for i, k in enumerate(pdf1.indice_figs)}
    pdf_med.indice_tabs = {k: 10 + i for i, k in enumerate(pdf1.indice_tabs)}
    pdf_med.indice_contenido()
    pdf_med.indice_figuras()
    pdf_med.indice_tablas()
    offset = pdf_med.page_no() - 1      # -1 = portada
    # El render final replica el borrador con `offset` páginas de índices de
    # más, así que el total es exactamente paginas_borrador + offset. Se fija
    # ANTES de dibujar para que el pie muestre "Página N de T" sin una 3ra
    # pasada de medición (y sin que el texto del pie altere el layout, porque
    # el pie se dibuja en la banda inferior).
    paginas_total = paginas_borrador + offset

    # ---------- RENDER #2: FINAL ----------
    pdf2 = ReporteCafeteria()
    pdf2.modo = "final"
    pdf2.total_pages = paginas_total
    pdf2.set_auto_page_break(auto=True, margin=15)
    pdf2.portada_institucional()
    # Índices con las páginas del borrador desplazadas por el offset.
    # El Resumen Ejecutivo va PRIMERO (front matter, sin número).
    pdf2.indice_caps = {"Resumen Ejecutivo": offset + 2}
    pdf2.indice_caps.update({k: v + offset for k, v in pdf1.indice_caps.items()})
    pdf2.indice_figs = {k: v + offset for k, v in pdf1.indice_figs.items()}
    pdf2.indice_tabs = {k: v + offset for k, v in pdf1.indice_tabs.items()}
    pdf2.indice_contenido()
    pdf2.indice_figuras()
    pdf2.indice_tablas()
    _resumen_ejecutivo(pdf2, ctx, textos, png, df)
    _construir_contenido(pdf2, ctx, textos, png, df)

    info = {
        "paginas_borrador": paginas_borrador,
        "paginas_indices": offset,
        "paginas_total": paginas_total,
        "paginas_reales": pdf2.page_no(),
        "offset": offset,
    }
    return pdf2, info


# ============================================================
# FRONT MATTER: RESUMEN EJECUTIVO (sin numeración)
# ============================================================
def _resumen_ejecutivo(pdf, ctx: dict, textos: dict, png: dict, df=None):
    """Dibuja el Resumen Ejecutivo como front matter, sin número.

    Va DESPUÉS de los 3 índices y ANTES del capítulo 1. No usa
    titulo_capitulo() porque no lleva numeración, pero SÍ se registra en el
    índice de contenido (clave sin número) con la página en la que arranca.
    """
    politica_records = ctx["politica_records"]
    contexto = ctx["contexto"]

    pdf.cap_actual = "Resumen Ejecutivo"        # encabezado de la página
    # Se registra la página en la que ARRANCARÁ el subtítulo (no add_page()
    # todavía: igual que titulo_capitulo, el registro va antes de abrirla).
    pdf.indice_caps["Resumen Ejecutivo"] = (pdf.page_no() or 0) + 1
    if pdf.page_no() > 0:
        pdf.add_page()
    pdf.subtitulo("Resumen Ejecutivo")

    pdf.nota_doble(textos["resumen"]["tecnico"], textos["resumen"]["simple"])
    pdf.subtitulo("KPIs Clave del Proyecto")
    politica_alta = next(
        (p["Mejor Acción"] for p in politica_records
         if p["Estado (Afluencia)"] == "Alta"), "N/A"
    )
    pdf.caption_tabla("Indicadores clave del proyecto")
    pdf.tabla_metrica([
        ("Registros en BD", f"{len(df)}"),
        ("Consumo promedio", f"${contexto['consumo_promedio_mxn']:.2f} MXN"),
        ("Satisfacción promedio", f"{contexto['satisfaccion_promedio']:.2f} / 5"),
        ("% Hora pico", f"{contexto['porcentaje_hora_pico']:.1f}%"),
        ("Mejor clasificador", contexto.get("clasificacion", {}).get("mejor_modelo", "N/A")),
        ("F1 del mejor modelo", f"{contexto.get('clasificacion', {}).get('mejor_f1', 'N/A')}"),
        ("Política óptima (afluencia alta)", politica_alta),
    ])
    pdf.preguntas_guia([
        "¿Los KPIs actuales coinciden con lo que percibe el personal de la cafetería?",
        "¿Qué decisión operativa NO cambiaría aunque empeoraran los KPIs?",
    ])
    pdf.decisiones_sugeridas(textos["dec_resumen"]["tecnico"])


# ============================================================
# CONTENIDO (capítulos) — dibujado en ambos renders
# ============================================================
def _construir_contenido(pdf, ctx: dict, textos: dict, png: dict, df=None):
    """Dibuja los 15 capítulos numerados sobre `pdf` (Fase 3B).

    Los 7 capítulos que eran esqueletos en 3A (2, 3, 5, 6, 7, 11 y 13) ya
    tienen contenido real, cada uno con su bloque de análisis IA. Las 8
    figuras y las 7 tablas clave se registran con caption_figura() /
    caption_tabla() para alimentar los índices de figuras y tablas.
    `df` solo se usa para `len(df)`.
    """
    res_reg = ctx["res_reg"]
    mejores = ctx["mejores"]
    res_km = ctx["res_km"]
    res_jer = ctx["res_jer"]
    politica = ctx["politica"]
    contexto = ctx["contexto"]
    desc = ctx["desc"]
    outliers_summary = ctx["outliers_summary"]

    # ---------- Nivel de muestra (Nivel 5) ----------
    n_actual = len(df) if df is not None else contexto["total_registros_bd"]
    _color_n, _nivel_n, _permite_ens, _permite_dl = _nivel_muestra(n_actual)

    # ---------- 1. DEFINICIÓN DEL PROBLEMA (antes "2. Introducción") ----------
    pdf.titulo_capitulo("1", "Definición del Problema")
    pdf.parrafo(
        "La cafetería universitaria presenta alta afluencia en horario matutino "
        "(7:00-11:00 AM), lo que genera cuellos de botella, insatisfacción del "
        "cliente y pérdida de ventas. La aplicación de técnicas de Aprendizaje "
        "Automático permite predecir el comportamiento, segmentar perfiles y "
        "optimizar la asignación de personal."
    )
    pdf.subtitulo("Tipos de problemas abordados")
    pdf.parrafo(
        "• Regresión: Predicción del Consumo Total (MXN).\n"
        "• Clasificación: Satisfacción Alta (4-5) vs Baja (1-3).\n"
        "• Agrupamiento: Segmentación con K-Means, Jerárquico y DBSCAN.\n"
        "• Refuerzo: Optimización de personal mediante Q-Learning."
    )

    # ---------- 2. PLANTEAMIENTO DE OBJETIVOS ----------
    pdf.titulo_capitulo("2", "Planteamiento de Objetivos")
    pdf.subtitulo("Objetivo General")
    pdf.parrafo(
        "Desarrollar un sistema funcional de Aprendizaje Automático que "
        "prediga el consumo, clasifique la satisfacción, segmente clientes y "
        "optimice la asignación de personal en la cafetería universitaria "
        "durante el horario matutino (7:00-11:00 AM), para apoyar la toma "
        "de decisiones operativas basadas en datos."
    )
    pdf.subtitulo("Objetivos Específicos")
    pdf.parrafo(
        "1. Preparación de datos: capturar, limpiar y validar registros en "
        "Supabase, detectando outliers por IQR y clasificando variables "
        "según su naturaleza estadística.\n"
        "2. Selección de modelos: comparar 2 modelos de regresión, 4 de "
        "clasificación, 3 de agrupamiento y 1 de refuerzo, justificando "
        "cada elección según el contexto del problema.\n"
        "3. Entrenamiento y optimización: entrenar con división 70/30 "
        "reproducible (random_state=42), ajustando hiperparámetros clave "
        "por modelo (max_depth, n_neighbors, k, alpha, gamma).\n"
        "4. Evaluación del desempeño: medir con MSE, R², F1, AUC-ROC y "
        "Silhouette; comparar y justificar el modelo ganador por tipo de "
        "problema.\n"
        "5. Implementación funcional: desplegar una app Streamlit que "
        "permita capturar datos, entrenar modelos y generar reportes PDF "
        "accionables."
    )
    pdf.subtitulo("Vinculación con el Temario del Curso")
    pdf.parrafo(
        "Cada objetivo específico corresponde a un paso del ciclo de vida "
        "de un proyecto de Aprendizaje Automático, cubriendo las 4 unidades "
        "del temario: fundamentos (U1), aprendizaje supervisado (U2), no "
        "supervisado (U3) y por refuerzo (U4). La app integra los cuatro "
        "en un mismo sistema funcional."
    )
    # Análisis IA (sin decisiones — es declarativo)
    pdf.nota_doble(
        textos["ana_objetivos"]["tecnico"],
        textos["ana_objetivos"]["simple"],
    )

    # ---------- 3. MARCO CONCEPTUAL ----------
    pdf.titulo_capitulo("3", "Marco Conceptual")
    pdf.subtitulo("Definición de Aprendizaje Automático aplicada al proyecto")
    pdf.parrafo(
        "Según Tom M. Mitchell (1997), un programa de computadora aprende "
        "de la experiencia E con respecto a una clase de tareas T y una "
        "medida de desempeño P, si su desempeño en T, medido por P, mejora "
        "con E. En este proyecto:\n"
        "• Tarea (T): predecir el consumo, clasificar la satisfacción, "
        "segmentar clientes y optimizar la asignación de personal.\n"
        "• Experiencia (E): registros históricos capturados en la "
        "cafetería durante el horario matutino.\n"
        "• Desempeño (P): métricas objetivas como MSE, R², F1, AUC-ROC "
        "y Silhouette."
    )
    pdf.subtitulo("Tipo de aprendizaje utilizado")
    pdf.parrafo(
        "• Supervisado: regresión (predicción de Consumo_Total_MXN) y "
        "clasificación (Satisfacción Alta vs Baja).\n"
        "• No supervisado: clustering (K-Means, Jerárquico, DBSCAN) para "
        "segmentación de clientes sin etiquetas.\n"
        "• Por refuerzo: Q-Learning tabular en un entorno simulado de "
        "asignación de baristas."
    )
    pdf.subtitulo("Conceptos del temario: aplicabilidad al proyecto")
    pdf.tabla_metrica([
        ("Aprendizaje supervisado", "Sí — regresión y clasificación"),
        ("Aprendizaje no supervisado", "Sí — clustering con K-Means"),
        ("Aprendizaje por refuerzo", "Sí — Q-Learning tabular"),
        ("Función objetivo", "Sí — MSE, Log-loss, Inercia, Q(s,a)"),
        ("Dataset y etiquetas", "Sí — 9 variables, 2 targets"),
        ("Clasificación de variables", "Sí — numéricas, nominales, ordinales"),
        ("Overfitting / Underfitting", "Sí — analizado en cap. 11"),
        ("Validación cruzada", "Parcial — cv=3 en SVM calibrado"),
        ("Deep Learning (redes neuronales)", "No — datos tabulares pequeños"),
        ("Transfer Learning", "No — sin modelo preentrenado para este dominio"),
        ("GANs", "No — no requiere generar datos realistas"),
    ], ancho1=70, ancho2=110,
        encabezados=("Concepto del temario", "¿Aplica al proyecto?"))
    pdf.subtitulo("Justificación de exclusiones")
    pdf.parrafo(
        "Deep Learning, Transfer Learning y GANs requieren datasets de "
        "miles o millones de registros, o datos no estructurados (imágenes, "
        f"texto, audio). Con {n_actual} registros tabulares (nivel {_color_n}: "
        f"{_nivel_n.lower()}), los modelos clásicos son la opción "
        "metodológicamente correcta: mejor interpretabilidad, menor riesgo de "
        "overfitting y menor costo computacional"
        + ("; con este N los ensembles tampoco se justifican."
           if not _permite_ens else
           "; aunque con este N los ensembles serían viables, el proyecto "
           "prioriza la interpretabilidad de los modelos clásicos.")
    )
    pdf.subtitulo("Función objetivo por tipo de problema")
    pdf.parrafo(
        "• Regresión: MSE (Error Cuadrático Medio).\n"
        "• Clasificación: Log-loss (cross-entropy) con regularización L2.\n"
        "• Clustering: Inercia (suma de distancias² al centroide).\n"
        "• Refuerzo: valor Q(s,a) actualizado por la Ecuación de Bellman."
    )
    # Análisis IA + preguntas (sin decisiones — es conceptual)
    pdf.nota_doble(
        textos["ana_marco"]["tecnico"],
        textos["ana_marco"]["simple"],
    )
    pdf.preguntas_guia([
        "¿El proyecto aplica correctamente el ciclo de vida de ML del temario?",
        "¿Falta algún concepto del curso que debería estar cubierto?",
        "¿La exclusión de Deep Learning está justificada metodológicamente?",
    ])

    # ---------- 4. ADQUISICIÓN Y ANÁLISIS DE DATOS (antes "3. Calidad") ----------
    pdf.titulo_capitulo("4", "Adquisición y Análisis de Datos")
    pdf.parrafo(
        f"El dataset contiene {len(df)} registros con 9 variables "
        f"({len(ctx['df_limpio'])} tras limpieza básica), capturados en horario matutino."
    )
    pdf.subtitulo("Estadísticas Básicas")
    filas = [
        (col, f"Media={float(desc.loc['mean', col]):.2f}, Std={float(desc.loc['std', col]):.2f}")
        for col in desc.columns
    ]
    pdf.caption_tabla("Estadísticas descriptivas de variables numéricas")
    pdf.tabla_metrica(filas)
    pdf.subtitulo("Detección de Outliers (IQR)")
    if not outliers_summary.empty:
        filas_out = [
            (row["Variable"],
             f"Q1={row['Q1']}, Q3={row['Q3']}, N={row['N outliers']} ({row['% outliers']}%)")
            for _, row in outliers_summary.iterrows()
        ]
        pdf.caption_tabla("Detección de outliers por método IQR")
        pdf.tabla_metrica(filas_out)
    pdf.nota_doble(textos["calidad"]["tecnico"], textos["calidad"]["simple"])
    pdf.preguntas_guia([
        "¿Los outliers corresponden a errores de captura o a comportamientos reales?",
        "¿Conviene eliminar los outliers o conservarlos porque representan casos extremos legítimos?",
    ])
    pdf.decisiones_sugeridas(textos["dec_calidad"]["tecnico"])

    # ---------- 5. PREPROCESAMIENTO DE DATOS ----------
    pdf.titulo_capitulo("5", "Preprocesamiento de Datos")
    pdf.subtitulo("Pipeline de preprocesamiento aplicado")
    pdf.tabla_metrica([
        ("Limpieza", "dropna en Consumo/Espera/Servicio; filtro Consumo<=250"),
        ("Normalización", "StandardScaler (media 0, std 1) sobre X_train"),
        ("Codificación", "LabelEncoder en Tipo_Cliente, Metodo_Pago, Es_Hora_Pico"),
        ("Manejo de nulos", "fillna(\"Nulo\") antes de LabelEncoder"),
        ("Reducción dim.", "PCA(n_components=2) solo para visualización"),
        ("Outliers", "Detección por IQR (1.5 * IQR)"),
    ], ancho1=55, ancho2=125)
    pdf.subtitulo("Justificación de las técnicas")
    pdf.parrafo(
        "• StandardScaler: obligatorio para KNN, SVM y distancias "
        "euclidianas; sin él, variables con rangos grandes (Consumo 0-500) "
        "dominarían sobre variables con rangos pequeños (Hora 7-11).\n"
        "• LabelEncoder: las variables categóricas tienen pocas categorías "
        "(3 por variable) y los modelos basados en árboles/KNN no se ven "
        "afectados por el orden numérico.\n"
        "• PCA: reduce dimensionalidad solo para visualización en 2D; el "
        "entrenamiento de clustering se hace sobre el espacio completo."
    )
    pdf.subtitulo("Técnicas NO aplicadas y su justificación")
    pdf.parrafo(
        "• One-Hot Encoding: no aplica porque con 3 categorías por variable "
        "el LabelEncoder es suficiente y evita aumentar dimensionalidad.\n"
        "• Imputación por media/mediana: no aplica, porque solo se filtran "
        "registros con NA en el target; las features con NA categórico "
        "reciben la etiqueta \"Nulo\".\n"
        "• t-SNE / UMAP: no aplican a n=200; son técnicas para datasets "
        "grandes que requieren más muestras para preservar estructura."
    )
    # Análisis IA + preguntas + decisiones
    pdf.nota_doble(
        textos["ana_preprocesamiento"]["tecnico"],
        textos["ana_preprocesamiento"]["simple"],
    )
    pdf.preguntas_guia([
        "¿Hay variables que requieran transformación adicional (ej. log)?",
        "¿El manejo actual de outliers puede estar eliminando casos reales?",
        "¿Conviene normalizar también las variables ordinales?",
    ])
    pdf.decisiones_sugeridas(textos["dec_preprocesamiento"]["tecnico"])

    # ---------- 6. DIVISIÓN DE DATOS ----------
    pdf.titulo_capitulo("6", "División de Datos")
    pdf.subtitulo("Esquema aplicado")
    pdf.parrafo(
        "Se emplea una división 70% entrenamiento / 30% prueba con "
        "train_test_split(random_state=42) para garantizar reproducibilidad. "
        "El conjunto de prueba se reserva exclusivamente para la evaluación "
        "final y NUNCA participa en el ajuste de hiperparámetros."
    )
    pdf.subtitulo("¿Por qué no 60/20/20 con conjunto de validación?")
    _n_val = max(int(round(n_actual * 0.20)), 1)
    pdf.parrafo(
        f"Con n={n_actual} registros (nivel {_color_n}), un conjunto de "
        f"validación separado (20% ≈ {_n_val} registros) "
        + ("sería estadísticamente insuficiente: las métricas tendrían "
           "varianza alta y no serían representativas. En su lugar, se usa "
           "validación cruzada interna (CalibratedClassifierCV con cv=3) en "
           "el modelo SVM, que es más robusto con muestras pequeñas. La "
           "rúbrica acepta 70/30 como opción válida."
           if n_actual < 500 else
           "sería viable en tamaño, pero se conserva la división 70/30 con "
           "validación cruzada interna (cv=3) para mantener comparabilidad y "
           "reservar el 30% de prueba. La rúbrica acepta 70/30 como opción "
           "válida.")
    )
    pdf.subtitulo("Tamaño mínimo idóneo de muestra")
    pdf.tabla_metrica([
        ("Para 70/30", "≥100 registros (≥30 en test)"),
        ("Para 60/20/20", "≥150 registros (≥30 en test)"),
        ("Para 5-fold CV", "≥250 registros (≥50 por fold)"),
        ("Para ensembles (RF/XGBoost)", "≥500 registros"),
        ("Para Deep Learning tabular", "≥1000 registros"),
        ("Recomendación del proyecto", "≥200 registros antes de generalizar"),
        ("Tu dataset actual",
         f"{n_actual} registros — nivel {_color_n}: {_nivel_n.lower()}"),
    ], ancho1=60, ancho2=120)
    pdf.subtitulo("Nota sobre la base de datos")
    pdf.parrafo(
        "La app NO altera la base de datos al generar el reporte; los "
        "modelos se entrenan en memoria sobre el DataFrame cargado. La "
        "simulación de Q-Learning es independiente y no usa datos de la BD. "
        "El generador sintético sí inserta registros en Supabase, pero solo "
        "cuando el usuario lo solicita explícitamente (con IDs nuevos "
        "forzados para evitar duplicados)."
    )
    # Análisis IA + preguntas (sin decisiones — es técnico)
    pdf.nota_doble(
        textos["ana_division"]["tecnico"],
        textos["ana_division"]["simple"],
    )
    pdf.preguntas_guia([
        "¿La muestra actual es suficiente para los 4 algoritmos comparados?",
        "¿Conviene implementar k-fold CV completa cuando n>500?",
        "¿El sesgo de selección por random_state=42 es despreciable?",
    ])

    # ---------- 7. SELECCIÓN DE MODELOS ----------
    pdf.titulo_capitulo("7", "Selección de Modelos")
    pdf.subtitulo("Modelos aplicados por tipo de problema")
    pdf.tabla_metrica([
        ("Regresión", "Lineal + Árbol de Regresión"),
        ("Clasificación", "Logística, KNN, SVM, Árbol"),
        ("Clustering", "K-Means, Jerárquico, DBSCAN"),
        ("Refuerzo", "Q-Learning tabular"),
    ], ancho1=50, ancho2=130)
    pdf.subtitulo("Modelos NO aplicados y su justificación")
    pdf.parrafo(
        "• Regresión polinómica: no se justifica con 7 features y relaciones "
        "aproximadamente lineales; añade complejidad sin mejora medible.\n"
        "• Naive Bayes: asume independencia entre features, irreal en "
        "contexto de cafetería (espera y servicio están correlacionados).\n"
        "• Random Forest / XGBoost: n<200 favorece modelos simples; "
        "ensembles requieren más datos para evitar overfitting.\n"
        "• Gaussian Mixture Models (GMM): asume distribuciones gaussianas "
        "que no se cumplen en datos mixtos (numéricos + categóricos).\n"
        "• Spectral Clustering: requiere matrices de similitud más grandes "
        "y es costoso computacionalmente para n=200.\n"
        "• DQN, Policy Gradient, A2C: el entorno de 3 estados × 3 acciones "
        "es demasiado simple; Q-Learning tabular es suficiente y más "
        "interpretable."
    )
    pdf.subtitulo("Regla de selección aplicada")
    if not _permite_ens:
        _cola_regla = (". Con este N, los ensembles y el Deep Learning no son "
                       "una opción válida por falta de datos.")
    elif not _permite_dl:
        _cola_regla = (". Con este N los ensembles serían técnicamente viables, "
                       "pero el proyecto se limita a modelos clásicos para "
                       "preservar interpretabilidad; el Deep Learning sigue "
                       "sin justificarse.")
    else:
        _cola_regla = (". Con este N tanto los ensembles como el Deep Learning "
                       "tabular serían técnicamente viables; el proyecto se "
                       "limita a modelos clásicos por decisión metodológica y "
                       "de interpretabilidad.")
    pdf.parrafo(
        "La complejidad del modelo debe ser proporcional al tamaño y "
        f"estructura del dataset. Con {n_actual} registros tabulares "
        f"(nivel {_color_n}: {_nivel_n.lower()}), los modelos clásicos son "
        "metodológicamente correctos: mejor interpretabilidad, menor riesgo "
        "de overfitting y menor costo computacional" + _cola_regla
    )
    pdf.subtitulo("Matriz de aplicabilidad")
    pdf.parrafo(
        f"La matriz resume qué técnicas aplican con el N real del dataset "
        f"({n_actual} registros, nivel {_color_n}) y qué se decidió en este "
        f"proyecto. La decisión se limita a modelos clásicos en todos los "
        f"casos, aun cuando la técnica sea viable en teoría."
    )
    pdf.tabla_metrica([
        ("Supervisado: regresión y clasificación",
         f"Sí — aplicado con {n_actual} registros"),
        ("No supervisado: K-Means, Jerárquico, DBSCAN",
         f"Sí — aplicado con {n_actual} registros"),
        ("Refuerzo: Q-Learning tabular",
         "Sí — aplicado sobre entorno simulado, sin datos de la BD"),
        ("Validación cruzada interna (cv=3)",
         "Sí — aplicada en el SVM calibrado"),
        ("Ensembles: Random Forest / XGBoost",
         "⚠ Viable desde n≥500 — no aplicado"
         + (" (N insuficiente)" if not _permite_ens
            else " (se prioriza interpretabilidad)")),
        ("Deep Learning tabular: MLP",
         "⚠ Viable desde n≥1000 — no aplicado"
         + (" (N insuficiente)" if not _permite_dl
            else " (se prioriza interpretabilidad)")),
        ("Transfer Learning y GANs",
         "✗ No aplica: sin datos no estructurados"),
    ], ancho1=90, ancho2=80)
    pdf.parrafo(
        f"Nota: N = {n_actual} registros (nivel {_color_n}). "
        "⚠ = técnica viable en teoría, pero no implementada en este "
        "proyecto; ✗ = técnica no aplicable al problema con cualquier N. "
        "El proyecto se limita a modelos clásicos en todos los casos."
    )
    # Análisis IA + preguntas + decisiones
    pdf.nota_doble(
        textos["ana_seleccion"]["tecnico"],
        textos["ana_seleccion"]["simple"],
    )
    pdf.preguntas_guia([
        "¿Los 4 clasificadores cubren el espectro de algoritmos del temario?",
        "¿Falta algún modelo clásico relevante (ej. LDA)?",
        "¿La exclusión de ensembles está justificada por el tamaño de muestra?",
    ])
    pdf.decisiones_sugeridas(textos["dec_seleccion"]["tecnico"])

    # ---------- 8. ENTRENAMIENTO SUPERVISADO ----------
    # Merge de "4. Regresión" + "5. Clasificación" en dos subsecciones.
    pdf.titulo_capitulo("8", "Entrenamiento y Evaluación Supervisada")

    # ---- 8.1 Regresión (antes "4. Modelos de Regresión") ----
    pdf.subtitulo("8.1 Regresión")
    pdf.parrafo("Predicción del Consumo Total (MXN) usando Regresión Lineal y Árboles.")
    if res_reg.get("error"):
        pdf.parrafo(f"Error en regresión: {res_reg['error']}")
    else:
        pdf.subtitulo("Métricas de Evaluación")
        m_lr, m_dt = res_reg["metricas_lr"], res_reg["metricas_dt"]
        pdf.caption_tabla("Métricas de evaluación de modelos de regresión")
        pdf.tabla_metrica([
            ("Regresión Lineal - MSE", f"{m_lr['MSE']:.2f}"),
            ("Regresión Lineal - R²", f"{m_lr['R2']:.4f} ({semaforo('R2', m_lr['R2'])})"),
            ("Árbol de Regresión - MSE", f"{m_dt['MSE']:.2f}"),
            ("Árbol de Regresión - R²", f"{m_dt['R2']:.4f} ({semaforo('R2', m_dt['R2'])})"),
        ])
        pdf.subtitulo("Ficha Técnica")
        pdf.parrafo(
            "• Modelos: Regresión Lineal (mínimos cuadrados) y DecisionTreeRegressor(max_depth=5).\n"
            "• Variables predictoras: Hora_Llegada, Tiempo_Espera_Min, Tiempo_Servicio_Min, "
            "Tipo_Cliente (encoding), Metodo_Pago (encoding), Es_Hora_Pico (encoding).\n"
            "• Target: Consumo_Total_MXN.\n"
            "• División: 70% entrenamiento / 30% prueba (random_state=42).\n"
            "• Escalado: StandardScaler sobre X_train; el mismo transform se aplica a X_test."
        )
        pdf.agregar_figura(png["reg"])
        pdf.caption_figura("Dispersión real vs predicho del modelo lineal")
        pdf.nota_doble(textos["reg"]["tecnico"], textos["reg"]["simple"])
        pdf.preguntas_guia([
            "¿Los puntos se ajustan a la diagonal o hay sesgo en consumos altos/bajos?",
            "Si el R² fuera 20% menor, ¿qué decisión operativa cambiaría?",
            "¿Qué variable adicional (clima, día de la semana, promociones) podría mejorar el modelo?",
        ])
        pdf.decisiones_sugeridas(textos["dec_reg"]["tecnico"])

    # ---- 8.2 Clasificación (antes "5. Modelos de Clasificación") ----
    pdf.subtitulo("8.2 Clasificación")
    pdf.parrafo("Predicción de Satisfacción Alta (4-5) vs Baja (1-3) con 4 algoritmos.")
    if mejores:
        pdf.subtitulo("Comparación de Modelos")
        pdf.caption_tabla("Comparación de clasificadores: F1 y Accuracy")
        filas = []
        for m, r in mejores.items():
            f1 = r["metricas"]["F1"]
            acc = r["metricas"]["Accuracy"]
            filas.append((m,
                f"F1={f1:.3f} ({semaforo('F1', f1)}), Acc={acc:.3f}"))
        pdf.tabla_metrica(filas, ancho1=55, ancho2=100)

        mejor_nombre = contexto["clasificacion"]["mejor_modelo"]
        mejor = mejores[mejor_nombre]
        pdf.parrafo(
            f"El mejor modelo fue {mejor_nombre} con F1-Score de "
            f"{contexto['clasificacion']['mejor_f1']:.3f}."
        )
        pdf.subtitulo("Ficha Técnica")
        hiperparams = {
            "Regresión Logística": "LogisticRegression(max_iter=1000, random_state=42)",
            "KNN": "KNeighborsClassifier(n_neighbors=5)",
            "SVM": "CalibratedClassifierCV(SVC(random_state=42), method='sigmoid', cv=3)",
            "Árbol de Decisión": "DecisionTreeClassifier(max_depth=5, random_state=42)",
        }.get(mejor_nombre, "N/A")
        pdf.parrafo(
            f"• Algoritmo ganador: {mejor_nombre} → {hiperparams}.\n"
            "• Variables predictoras: las mismas 7 del modelo de regresión.\n"
            "• Target: Satisfaccion_Alta = 1 si Satisfaccion_1_al_5 ≥ 4, si no 0.\n"
            "• División: 70% / 30% (random_state=42).\n"
            "• Escalado: StandardScaler sobre X_train."
        )
        pdf.agregar_figura(png["cm"])
        pdf.caption_figura(f"Matriz de confusión del modelo {mejor_nombre}")
        if png["roc"]:
            pdf.agregar_figura(png["roc"])
            pdf.caption_figura(f"Curva ROC del modelo {mejor_nombre}")
        pdf.nota_doble(textos["clf"]["tecnico"], textos["clf"]["simple"])
        pdf.preguntas_guia([
            "¿Qué es más costoso para la cafetería: un falso positivo o un falso negativo de insatisfacción?",
            "¿Cuál sería el costo operativo de predecir satisfacción alta cuando en realidad es baja?",
            "¿Se justifica la complejidad de SVM si Logística da un F1 similar?",
        ])
        pdf.decisiones_sugeridas(textos["dec_clf"]["tecnico"])

    # ---------- 9. NO SUPERVISADO (antes "6.") ----------
    pdf.titulo_capitulo("9", "Entrenamiento y Evaluación No Supervisada")
    pdf.parrafo("Segmentación de clientes con K-Means, Jerárquico y DBSCAN.")
    if not res_km.get("error"):
        sil = res_km["silhouette"]
        pdf.subtitulo("K-Means (k=3)")
        pdf.parrafo(
            f"Índice de Silhouette: {sil:.4f} ({semaforo('Silhouette', sil)})."
        )
        pdf.subtitulo("Ficha Técnica")
        pdf.parrafo(
            "• Algoritmo: KMeans(n_clusters=3, n_init=10, random_state=42).\n"
            "• Features estandarizadas: 7 variables (numéricas + categóricas con LabelEncoder).\n"
            "• Reducción para visualización: PCA a 2 componentes (solo para graficar).\n"
            "• Validación: silhouette_score sobre el espacio escalado completo."
        )
        pdf.agregar_figura(png["km_codo"])
        pdf.caption_figura("Selección de k mediante método del codo")
        pdf.agregar_figura(png["km_pca"])
        pdf.caption_figura("Visualización de clústeres K-Means en PCA 2D")
        pdf.nota_doble(textos["km"]["tecnico"], textos["km"]["simple"])
        pdf.preguntas_guia([
            "¿Los clústeres corresponden a tipos de cliente reales o son artefactos del algoritmo?",
            "¿Qué acción de negocio diferenciada aplicaría a cada clúster?",
            "¿Cuántos clústeres son accionables operativamente (no solo matemáticamente)?",
        ])
        pdf.decisiones_sugeridas(textos["dec_km"]["tecnico"])

    # ---------- 6.5 JERÁRQUICO (antes huérfano) ----------
    if not res_jer.get("error"):
        pdf.subtitulo("Clustering Jerárquico (Aglomerativo)")
        pdf.parrafo(
            "• Algoritmo: AgglomerativeClustering con linkage='ward'.\n"
            "• Distancia: euclidiana sobre features estandarizadas.\n"
            "• Visualización: dendrograma hasta hoja completa."
        )
        pdf.agregar_figura(png["jer"])
        pdf.caption_figura("Dendrograma del clustering jerárquico aglomerativo")
        pdf.nota_doble(
            textos["ana_dendro"]["tecnico"],
            textos["ana_dendro"]["simple"],
        )
        pdf.preguntas_guia([
            "¿En qué altura del dendrograma se aprecia el corte más natural?",
            "¿Las ramas más largas corresponden a clientes atípicos o a un clúster pequeño?",
            "¿El resultado jerárquico coincide con los clústeres de K-Means?",
        ])
        pdf.decisiones_sugeridas(textos["dec_dendro"]["tecnico"])

    # ---------- 10. REFUERZO (antes "7.") ----------
    pdf.titulo_capitulo("10", "Entrenamiento y Evaluación por Refuerzo")
    pdf.parrafo(
        "Se implementó Q-Learning en un entorno simulado donde un agente (barista) "
        "decide cuántos empleados asignar según la afluencia (Baja, Media, Alta)."
    )
    pdf.agregar_figura(png["ap"])
    pdf.caption_figura("Curva de aprendizaje del agente Q-Learning")
    pdf.agregar_figura(png["q"])
    pdf.caption_figura("Matriz Q final: valor por estado y acción")
    pdf.subtitulo("Ficha Técnica")
    pdf.parrafo(
        "• Algoritmo: Q-Learning tabular, política ε-greedy (ε=0.1).\n"
        "• Hiperparámetros: α=0.1, γ=0.9, episodios=500, max_pasos=50 por episodio.\n"
        "• Estados: 3 niveles de afluencia (Baja / Media / Alta).\n"
        "• Acciones: 3 niveles de staffing (1 / 2 / 3 baristas).\n"
        "• Recompensa: matriz R(estado, acción) del simulador (ganancia estimada).\n"
        "• Transiciones: probabilísticas (simulan variabilidad de la afluencia)."
    )
    pdf.subtitulo("Política Óptima Aprendida")
    for _, row in politica.iterrows():
        pdf.parrafo(f"• Estado {row['Estado (Afluencia)']}: {row['Mejor Acción']} "
                    f"(Q={row['Valor Q']})")
    pdf.nota_doble(textos["rl"]["tecnico"], textos["rl"]["simple"])
    pdf.preguntas_guia([
        "¿Cómo cambiaría la política si la recompensa priorizara satisfacción sobre ganancia?",
        "¿Qué tan sensible es la política óptima a los valores de α y γ?",
        "¿Cómo se validaría esta política contra datos reales antes de implantarla?",
    ])
    pdf.decisiones_sugeridas(textos["dec_rl"]["tecnico"])

    # ---------- 11. VALIDACIÓN Y ANÁLISIS DE RESULTADOS ----------
    pdf.titulo_capitulo("11", "Validación y Análisis de Resultados")

    # ---- 11.1 Overfitting ----
    pdf.subtitulo("11.1 Detección de Overfitting")
    pdf.parrafo(
        "Se compara el desempeño del modelo en entrenamiento (train) contra "
        "prueba (test). Un delta > 0.15 en R² o F1 sugiere posible "
        "overfitting (el modelo memoriza en lugar de generalizar)."
    )
    filas_ov = []
    if not res_reg.get("error"):
        for nombre_ov, k_test, k_train in [
            ("Regresión Lineal", "metricas_lr", "metricas_lr_train"),
            ("Árbol de Regresión", "metricas_dt", "metricas_dt_train"),
        ]:
            r2_train = float(res_reg[k_train]["R2"])
            r2_test = float(res_reg[k_test]["R2"])
            delta_r2 = r2_train - r2_test
            filas_ov.append((
                nombre_ov,
                f"R² train={r2_train:.3f} | test={r2_test:.3f} | "
                f"Δ={delta_r2:+.3f} ({'OVERFITTING' if delta_r2 > 0.15 else 'OK'})",
            ))
    for nombre_ov, r_ov in mejores.items():
        f1_train = float(r_ov["metricas_train"]["F1"])
        f1_test = float(r_ov["metricas"]["F1"])
        delta_f1 = f1_train - f1_test
        filas_ov.append((
            nombre_ov,
            f"F1 train={f1_train:.3f} | test={f1_test:.3f} | "
            f"Δ={delta_f1:+.3f} ({'OVERFITTING' if delta_f1 > 0.15 else 'OK'})",
        ))
    if filas_ov:
        pdf.caption_tabla("Comparación train vs test para detección de overfitting")
        pdf.tabla_metrica(filas_ov, ancho1=60, ancho2=120)

    # ---- 11.2 Underfitting (diagnóstico con los valores reales del run) ----
    pdf.subtitulo("11.2 Detección de Underfitting")
    mejor_r2_test = None
    if not res_reg.get("error"):
        mejor_r2_test = max(
            float(res_reg["metricas_lr"]["R2"]),
            float(res_reg["metricas_dt"]["R2"]),
        )
    mejor_f1_test = (
        max(float(m["metricas"]["F1"]) for m in mejores.values())
        if mejores else None
    )

    if mejor_r2_test is None:
        diag_reg = "sin datos de regresión (N/A)."
    elif mejor_r2_test < 0.3:
        diag_reg = "SÍ se observa underfitting en regresión (R² < 0.3)."
    else:
        diag_reg = "NO se observa underfitting severo en regresión."

    if mejor_f1_test is None:
        diag_clf = "sin datos de clasificación (N/A)."
    elif mejor_f1_test < 0.5:
        diag_clf = "SÍ se observa underfitting en clasificación (F1 < 0.5)."
    else:
        diag_clf = "NO se observa underfitting severo en clasificación."

    txt_r2 = f"{mejor_r2_test:.3f}" if mejor_r2_test is not None else "N/A"
    txt_f1 = f"{mejor_f1_test:.3f}" if mejor_f1_test is not None else "N/A"
    pdf.parrafo(
        "R² < 0.3 en regresión o F1 < 0.5 en clasificación indican modelo "
        "demasiado simple para capturar la estructura del problema. Con los "
        "datos actuales:\n"
        f"• Mejor R² (test) = {txt_r2} → {diag_reg}\n"
        f"• Mejor F1 (test) = {txt_f1} → {diag_clf}"
    )

    # ---- 11.3 Mitigaciones ----
    pdf.subtitulo("11.3 Mitigaciones aplicadas")
    pdf.parrafo(
        "• Poda preventiva: max_depth=5 en árboles para limitar profundidad.\n"
        "• Regularización: L2 por defecto en Regresión Logística.\n"
        "• Validación cruzada: CalibratedClassifierCV(cv=3) en SVM.\n"
        "• División 70/30: conjunto de prueba nunca participa en ajuste."
    )

    # ---- 11.4 Conclusiones y Limitaciones (texto de v3.3) ----
    pdf.subtitulo("11.4 Conclusiones y Limitaciones")
    pdf.nota_doble(textos["conclusiones"]["tecnico"], textos["conclusiones"]["simple"])
    _texto_nivel = {
        "ROJO": "El tamaño de muestra es insuficiente para generalizar: los "
                "resultados son indicativos y deben leerse como evidencia "
                "preliminar.",
        "AMARILLO": "El tamaño de muestra es preliminar: conviene ampliar a 200 "
                    "o más registros antes de generalizar a todos los turnos.",
        "VERDE": "El tamaño de muestra es suficiente para obtener conclusiones "
                 "estables con modelos clásicos, sin recurrir a arquitecturas "
                 "profundas.",
        "VERDE+": "El tamaño de muestra es excelente y permitiría ensembles; "
                  "el proyecto se limita a modelos clásicos por "
                  "interpretabilidad.",
        "VERDE++": "El tamaño de muestra es excelente y permitiría ensembles y "
                   "Deep Learning tabular; el proyecto mantiene modelos "
                   "clásicos por decisión metodológica.",
    }
    tam_muestra_nota = (
        f"El dataset cuenta con {n_actual} registros tabulares (nivel "
        f"{_color_n}: {_nivel_n.lower()}). {_texto_nivel[_color_n]}"
    )
    pdf.parrafo(
        f"• {tam_muestra_nota}\n"
        "• El Aprendizaje por Refuerzo opera sobre un entorno simulado y no sobre telemetría real.\n"
        "• No se aplicaron Transfer Learning ni GANs debido a la naturaleza tabular."
    )

    # Análisis IA + preguntas + decisiones del capítulo
    pdf.nota_doble(
        textos["ana_validacion"]["tecnico"],
        textos["ana_validacion"]["simple"],
    )
    pdf.preguntas_guia([
        "¿El delta train/test observado indica overfitting preocupante?",
        "¿La muestra actual permite estimaciones estables de F1/AUC?",
        "¿Qué modelo conviene desplegar si se prioriza generalización?",
    ])
    pdf.decisiones_sugeridas(textos["dec_validacion"]["tecnico"])

    # ---------- 12. IMPLEMENTACIÓN FUNCIONAL ----------
    pdf.titulo_capitulo("12", "Implementación Funcional")

    # ---- 12.1 Descripción de la aplicación ----
    pdf.subtitulo("12.1 Descripción de la Aplicación Streamlit")
    pdf.parrafo(
        "La implementación funcional del proyecto se materializa en una "
        "aplicación web desarrollada con Streamlit y desplegada en Streamlit "
        "Community Cloud. La app permite: (1) capturar registros manualmente, "
        "(2) cargar archivos CSV/Excel, (3) generar datos sintéticos, (4) "
        "entrenar modelos supervisados, no supervisados y de refuerzo, y (5) "
        "descargar reportes PDF con análisis asistido por IA. El código fuente "
        "está versionado en GitHub y la base de datos se aloja en Supabase "
        "(PostgreSQL en la nube)."
    )

    # ---- 12.2 Plan de acción (antes "8. Plan") ----
    pdf.subtitulo("12.2 Plan de Acción Operativo")
    pdf.nota_doble(textos["plan"]["tecnico"], textos["plan"]["simple"])

    # ---- 12.3 Riesgos (antes "9. Riesgos") ----
    pdf.subtitulo("12.3 Riesgos y Próximos Pasos")
    pdf.nota_doble(textos["riesgos"]["tecnico"], textos["riesgos"]["simple"])
    pdf.decisiones_sugeridas(textos["dec_riesgos"]["tecnico"])

    # ---------- 13. ENTREGABLES ----------
    pdf.titulo_capitulo("13", "Entregables del Proyecto")
    pdf.parrafo(
        "Conforme a la rúbrica del Proyecto Final, los entregables son:"
    )
    pdf.subtitulo("13.1 Presentación")
    pdf.parrafo(
        "Documento ejecutivo con portada, índice, introducción, descripción "
        "del problema, objetivos, metodología, modelo utilizado, resultados, "
        "conclusiones y referencias. Este reporte PDF constituye la base "
        "metodológica de la presentación."
    )
    pdf.subtitulo("13.2 Video demostrativo")
    pdf.parrafo(
        "Grabación de 8-12 minutos mostrando la app funcionando en vivo: "
        "captura de datos, entrenamiento de modelos, generación de reportes "
        "y análisis de resultados. Participación de todos los integrantes "
        "con rostro visible, subido a YouTube."
    )
    pdf.subtitulo("13.3 Código funcional")
    pdf.parrafo(
        "Repositorio GitHub con la app Streamlit completa:\n"
        "• App en producción: https://cafeteria-ml-aprendizaje-auto.streamlit.app\n"
        "• Repositorio: https://github.com/GGRAFSS/cafeteria-ml\n"
        "• Base de datos: Supabase (PostgreSQL en la nube)\n"
        "• Reproducibilidad: random_state=42, seed=42, requirements.txt versionado"
    )
    pdf.subtitulo("13.4 Cobertura de la Rúbrica")
    pdf.caption_tabla("Cobertura de los 12 puntos de la rúbrica")
    pdf.tabla_metrica([
        ("1. Definición del problema", "Cap. 1 + app"),
        ("2. Objetivos", "Cap. 2"),
        ("3. Marco conceptual", "Cap. 3"),
        ("4. Análisis de datos", "Cap. 4 + app"),
        ("5. Preprocesamiento", "Cap. 5"),
        ("6. División de datos", "Cap. 6"),
        ("7. Selección del modelo", "Cap. 7"),
        ("8. Entrenamiento", "Cap. 8, 9, 10"),
        ("9. Evaluación", "Cap. 8, 9, 10"),
        ("10. Validación y análisis", "Cap. 11"),
        ("11. Implementación funcional", "Cap. 12 + app"),
        ("12. Entregables", "Este capítulo + repo + video"),
    ], ancho1=80, ancho2=100,
        encabezados=("Punto de la rúbrica", "Dónde se cubre"))
    # Sin análisis IA (es informativo)

    # ---------- 14. GLOSARIO (antes "10.5") ----------
    pdf.titulo_capitulo("14", "Glosario")
    pdf.parrafo(
        "Definiciones para lectores sin formación técnica en ciencia de datos."
    )
    glosario = [
        ("R² (Coeficiente de determinación)",
         "Qué porcentaje de la variación de una variable explica el modelo. "
         "Va de 0 a 1; más cerca de 1 = mejor."),
        ("MSE (Error Cuadrático Medio)",
         "Promedio del cuadrado de los errores de predicción. Más bajo = mejor. "
         "Su raíz cuadrada se interpreta en pesos (MXN)."),
        ("Accuracy / Exactitud",
         "Porcentaje de predicciones correctas sobre el total."),
        ("Precision / Precisión",
         "De los que el modelo dijo que eran de una clase, cuántos realmente lo eran."),
        ("Recall / Exhaustividad",
         "De los que realmente eran de una clase, cuántos detectó el modelo."),
        ("F1-Score",
         "Promedio armónico entre precisión y exhaustividad. Arriba de 0.7 suele ser aceptable."),
        ("AUC-ROC",
         "Capacidad para separar clases. 0.5 = azar, 1.0 = perfecto."),
        ("Matriz de Confusión",
         "Tabla que cruza predicciones con valores reales."),
        ("Silhouette",
         "Qué tan bien separados están los clústeres. Arriba de 0.5 indica estructura clara."),
        ("PCA (Análisis de Componentes Principales)",
         "Técnica para reducir muchas variables a 2 o 3 dimensiones para visualizar."),
        ("Dendrograma",
         "Diagrama de árbol que muestra cómo se agrupan los datos en el clustering jerárquico."),
        ("Q-Learning / Ecuación de Bellman",
         "Algoritmo de aprendizaje por refuerzo donde el agente aprende el valor de cada acción."),
        ("Outlier",
         "Valor atípico que se aleja mucho del resto."),
        ("Train/Test Split",
         "División del dataset: 70% entrenar, 30% evaluar."),
        ("StandardScaler / Escalado",
         "Transformación que centra cada variable en media 0 y desviación 1."),
        ("LabelEncoder / Encoding",
         "Convertir categorías de texto a números."),
        ("Overfitting",
         "Cuando el modelo memoriza los datos de entrenamiento pero falla con datos nuevos."),
        ("Hiperparámetro",
         "Configuración elegida ANTES de entrenar (ej. max_depth, k, alpha)."),
    ]
    pdf.tabla_larga(glosario)

    # ---------- 15. REFERENCIAS (antes "11.") ----------
    pdf.titulo_capitulo("15", "Referencias")
    pdf.parrafo(
        "1. Mitchell, T. (1997). Machine Learning. McGraw-Hill.\n"
        "2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.\n"
        "3. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.\n"
        "4. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly.\n"
        "5. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12."
    )


# ============================================================
# REPORTE RESUMEN (1 PÁGINA)
# ============================================================
def generar_resumen_pdf(df: pd.DataFrame) -> tuple:
    if df.empty or len(df) < 20:
        return None, "Se necesitan al menos 20 registros para generar el resumen."

    try:
        ctx = _construir_contexto(df)
        contexto = ctx["contexto"]
        resumen_exec = ctx["resumen_exec"]

        fallback = (
            f"{len(df)} registros; consumo promedio ${resumen_exec['consumo_prom']:.2f}; "
            f"satisfacción {resumen_exec['sat_prom']:.2f}/5; "
            f"mejor clasificador: "
            f"{contexto.get('clasificacion', {}).get('mejor_modelo', 'N/A')}."
        )
        textos = analizar_seccion_ia(
            "Resumen Ejecutivo (versión 1 página)",
            contexto,
            fallback,
            instruccion_extra=(
                "Resumen ejecutivo MUY conciso (máx 110 palabras) que cubra: "
                "volumen de datos, consumo promedio, satisfacción, mejor modelo, "
                "política óptima de baristas y 2 decisiones operativas clave."
            ),
        )

        pdf = ReporteCafeteria()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.cap_actual = "Resumen Ejecutivo"   # título del encabezado de página
        pdf.add_page()

        pdf.set_font(pdf.FONT, "B", 16)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(0, 10, "Cafetería Universitaria — Resumen Ejecutivo", 0, 1, "C")
        pdf.set_font(pdf.FONT, "", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, "Aprendizaje Automático · Maestría en Inteligencia Artificial · TecNM", 0, 1, "C")
        pdf.cell(0, 5, f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1, "C")
        pdf.ln(4)

        pdf.subtitulo("Panorama General")
        pdf.parrafo(textos["tecnico"])
        if textos["simple"]:
            pdf.set_text_color(20, 90, 50)
            pdf.parrafo(textos["simple"])
            pdf.set_text_color(0, 0, 0)

        pdf.subtitulo("Indicadores Clave")
        politica_alta = next(
            (p["Mejor Acción"] for p in ctx["politica_records"]
             if p["Estado (Afluencia)"] == "Alta"), "N/A"
        )
        pdf.tabla_metrica([
            ("Registros en BD", f"{len(df)}"),
            ("Consumo promedio", f"${contexto['consumo_promedio_mxn']:.2f} MXN"),
            ("Consumo total observado", f"${contexto['consumo_total_mxn']:.2f} MXN"),
            ("Satisfacción promedio", f"{contexto['satisfaccion_promedio']:.2f} / 5"),
            ("% Hora pico", f"{contexto['porcentaje_hora_pico']:.1f}%"),
            ("Espera promedio", f"{contexto['espera_promedio_min']:.1f} min"),
            ("Mejor clasificador", contexto.get("clasificacion", {}).get("mejor_modelo", "N/A")),
            ("F1 del mejor modelo", f"{contexto.get('clasificacion', {}).get('mejor_f1', 'N/A')}"),
            ("Política óptima (afluencia alta)", politica_alta),
        ], ancho1=70, ancho2=100)

        pdf.subtitulo("Acciones Prioritarias")
        politica_txt = ", ".join(
            f"{p['Estado (Afluencia)']}→{p['Mejor Acción']}"
            for p in ctx["politica_records"]
        )
        pdf.parrafo(
            f"1. Asignación de personal según afluencia (política Q-Learning): {politica_txt}.\n"
            "2. Agilizar caja en hora pico con prepago por app.\n"
            "3. Segmentar promociones por tipo de cliente y horario.\n"
            "4. Revisar outliers de espera (>25 min) y consumo (>200 MXN).\n"
            "5. Ampliar el dataset a >200 registros para estabilizar modelos."
        )

        pdf.ln(4)
        pdf.set_font(pdf.FONT, "I", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.multi_cell(0, 4,
            "Este resumen es un extracto del reporte completo. "
            "Solicite el documento íntegro para ver metodología, gráficas, "
            "fichas técnicas, comparativas críticas y glosario."
        )

        return bytes(pdf.output()), None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error al generar resumen: {e}"


# ============================================================
# FASE 5A + 5B: REPORTE DIDÁCTICO (estructura y contenido)
# ============================================================
# 4 workers, mismo balance que _ejecutar_ia_paralelo (velocidad vs rate limit).
_FICHA_WORKERS = 4

# Instrucción base: fija el rol y evita que el modelo cambie el formato.
_INSTRUCCION_FICHA = (
    "Actúas como docente de Aprendizaje Automático que explica a estudiantes "
    "de posgrado un objeto concreto del reporte de la cafetería. Describe solo "
    "ese objeto, con los valores reales del payload, sin inventar cifras y sin "
    "repetir el contenido de otras fichas. El bloque 'no_aplica' debe nombrar "
    "técnicas concretas del temario que no sirven para este objeto y dar la "
    "razón técnica (tipo de variable, supuestos incumplidos, objetivo del "
    "análisis o costo computacional)."
)


def _d(valor, formato="{:.2f}", defecto="N/D") -> str:
    """Formatea un número del contexto; usa `defecto` si falta o no es numérico."""
    try:
        return formato.format(float(valor))
    except (TypeError, ValueError):
        return defecto


def _ficha_curada(titulo: str, referencia: str, datos: dict,
                  fallback: dict, foco: str) -> dict:
    """Empaqueta una ficha: metadatos del render, payload real y respaldo curado.

    `fallback` trae las siete claves de CAMPOS_DIDACTICOS redactadas con los
    valores del run, de modo que el reporte nunca queda en blanco si la IA no
    está disponible o responde mal.
    """
    return {
        "titulo": titulo,
        "referencia": referencia,
        "datos": datos,
        "fallback": fallback,
        "instruccion": f"{_INSTRUCCION_FICHA}\nFoco de esta ficha: {foco}",
    }


def _especificaciones_didacticas(ctx: dict, df=None) -> list[dict]:
    """Arma las 15 fichas (8 figuras + 7 tablas) con su payload y su respaldo.

    Cada especificación trae: `titulo` y `referencia` (lo que dibuja el
    render), `datos` (métricas reales que recibe la IA), `fallback` (texto
    curado por bloque con los valores del run) e `instruccion` (enfoque).
    """
    ctx = ctx or {}
    contexto = ctx.get("contexto", {}) or {}
    res_reg = ctx.get("res_reg", {}) or {}
    mejores = ctx.get("mejores", {}) or {}
    res_km = ctx.get("res_km", {}) or {}
    politica_records = ctx.get("politica_records", []) or []
    desc = ctx.get("desc")
    outliers_por_var = contexto.get("outliers_por_variable", []) or []

    n_bd = contexto.get("total_registros_bd", len(df) if df is not None else 0)
    n_limpio = contexto.get("registros_limpios", n_bd)
    n_prueba = max(int(round(n_bd * 0.30)), 1)
    n_out = contexto.get("outliers_detectados", 0)
    pct_out = round(n_out / max(n_bd, 1) * 100, 2)
    consumo_prom = contexto.get("consumo_promedio_mxn")
    consumo_total = contexto.get("consumo_total_mxn")
    espera_prom = contexto.get("espera_promedio_min")
    espera_max = contexto.get("espera_maxima_min")
    servicio_prom = contexto.get("servicio_promedio_min")
    pct_pico = contexto.get("porcentaje_hora_pico")
    sat_prom = contexto.get("satisfaccion_promedio")

    reg = contexto.get("regresion", {}) or {}
    clf = contexto.get("clasificacion", {}) or {}
    km = contexto.get("clustering_kmeans", {}) or {}
    rl = contexto.get("refuerzo_q_learning", {}) or {}

    mejor = clf.get("mejor_modelo", "el mejor modelo")
    mejor_f1 = clf.get("mejor_f1")
    mejor_auc = clf.get("mejor_auc")
    modelos = clf.get("modelos", {}) or {}
    met_mejor = (mejores.get(mejor) or {}).get("metricas", {}) if mejores else {}
    sil = km.get("silhouette")
    rl_ini = rl.get("recompensa_prom_primeros_50")
    rl_fin = rl.get("recompensa_prom_ultimos_50")
    episodios = rl.get("episodios", 500)

    politica_txt = ", ".join(
        f"{p.get('Estado (Afluencia)', '?')}→{p.get('Mejor Acción', '?')}"
        for p in politica_records
    ) or "N/D"
    politica_alta = next(
        (p.get("Mejor Acción") for p in politica_records
         if p.get("Estado (Afluencia)") == "Alta"), "N/D")

    # Mejor regresor según R² de prueba (para la tabla 4 y la figura 1)
    mejor_reg = "N/D"
    try:
        lr_r2, dt_r2 = float(reg.get("lr_r2")), float(reg.get("dt_r2"))
        mejor_reg = "Regresión Lineal" if lr_r2 >= dt_r2 else "Árbol de Regresión"
    except (TypeError, ValueError):
        lr_r2, dt_r2 = reg.get("lr_r2"), reg.get("dt_r2")

    # El signo del R² cambia la lectura: negativo = peor que el promedio
    try:
        if lr_r2 is None:
            ajuste_lr = "indeterminado"
        elif float(lr_r2) < 0:
            ajuste_lr = (
                "negativo, es decir, peor que pronosticar con el promedio del "
                "consumo en todos los casos"
            )
        elif float(lr_r2) < 0.30:
            ajuste_lr = "bajo: apenas mejora el pronóstico del promedio"
        else:
            ajuste_lr = "moderado para datos de consumo con pocas variables"
    except (TypeError, ValueError):
        ajuste_lr = "indeterminado"

    # Estadísticas descriptivas reales (tabla 2)
    est_desc = {}
    if isinstance(desc, pd.DataFrame) and not desc.empty:
        for col in desc.columns:
            est_desc[col] = {
                "media": round(float(desc.loc["mean", col]), 2),
                "desviacion": round(float(desc.loc["std", col]), 2),
                "minimo": round(float(desc.loc["min", col]), 2),
                "maximo": round(float(desc.loc["max", col]), 2),
            }

    def _media(col):
        try:
            return _d(float(desc.loc["mean", col]), "{:.2f}")
        except Exception:
            return "N/D"

    def _std(col):
        try:
            return _d(float(desc.loc["std", col]), "{:.2f}")
        except Exception:
            return "N/D"

    fichas = []

    # ---------- D1 · Figura 1: dispersión real vs predicho ----------
    fichas.append(_ficha_curada(
        "Figura 1. Dispersión real vs predicho del modelo lineal",
        "Figura 1 del reporte completo · Unidad 1 (regresión lineal)",
        {
            "objeto": "Figura 1 del reporte completo",
            "tipo_grafico": "dispersión real vs predicho con línea de predicción perfecta",
            "modelo": "Regresión Lineal (mínimos cuadrados)",
            "target": "Consumo_Total_MXN",
            "registros_totales": n_bd,
            "registros_en_prueba": n_prueba,
            "R2_prueba_lineal": reg.get("lr_r2"),
            "MSE_prueba_lineal": reg.get("lr_mse"),
            "R2_prueba_arbol": reg.get("dt_r2"),
            "MSE_prueba_arbol": reg.get("dt_mse"),
            "consumo_promedio_mxn": consumo_prom,
        },
        {
            "que_se_ve": (
                f"Gráfico de dispersión con un punto por cada registro del "
                f"conjunto de prueba (≈{n_prueba} de {n_bd}). El eje horizontal "
                f"muestra el consumo real en pesos y el vertical el consumo "
                f"predicho por la regresión lineal; una diagonal punteada marca "
                f"la predicción perfecta. La nube se agrupa alrededor de esa "
                f"diagonal, con dispersión vertical apreciable en consumos "
                f"medios y algunos puntos alejados en consumos altos."
            ),
            "tecnica": (
                f"Con R² = {_d(reg.get('lr_r2'), '{:.4f}')} y MSE = "
                f"{_d(reg.get('lr_mse'))} MXN² en prueba, el ajuste de la "
                f"regresión lineal es {ajuste_lr}. El consumo promedio es "
                f"${_d(consumo_prom)} MXN y el Árbol de Regresión alcanza "
                f"R² = {_d(reg.get('dt_r2'), '{:.4f}')} con MSE = "
                f"{_d(reg.get('dt_mse'))}: {mejor_reg} es el que mejor se acerca "
                f"a la diagonal. La lectura operativa no cambia con el signo del "
                f"R²: mientras más compacta la nube alrededor de la diagonal, más "
                f"útil es el pronóstico."
            ),
            "simple": (
                "Cada punto es un cliente: a la derecha está lo que realmente "
                "gastó y hacia arriba lo que el modelo calculó. Si el modelo "
                "fuera perfecto, todos los puntos caerían sobre la línea "
                "punteada. Aquí la nube la rodea, pero abierta: alcanza para "
                "ordenar clientes y anticipar tendencias, no para adivinar el "
                "gasto exacto de una persona."
            ),
            "temario": (
                "Unidad 1 — Regresión lineal: la gráfica real vs predicho es el "
                "diagnóstico estándar de ajuste; R² cuantifica la varianza "
                "explicada y el MSE el error cuadrático medio en pesos."
            ),
            "pedagogia": (
                "Enseña a leer un regresor por la forma de su error y no por una "
                "sola cifra: la nube revela sesgo (nube desplazada), "
                "heterocedasticidad (abanico que se abre) y valores atípicos "
                "(puntos lejanos), tres diagnósticos que ningún valor de R² "
                "resume por sí solo."
            ),
            "errores": (
                "• Interpretar R² como porcentaje de aciertos del modelo. "
                "• Suponer que un R² bajo invalida el análisis, cuando con "
                "consumo humano y pocas variables es lo esperado. "
                "• Leer la nube como si describiera clientes individuales en "
                "lugar de una tendencia promedio."
            ),
            "no_aplica": (
                "• Exactitud, matriz de confusión y F1: el objetivo es continuo "
                "(pesos), no hay clases que contar. • K-Means, PCA o silhouette: "
                "segmentar residuales no aporta lectura operativa y diluye el "
                "diagnóstico de ajuste. • Q-Learning: no existe agente ni "
                "secuencia de decisiones que aprender. • Regresión logística: el "
                "target no es binario."
            ),
        },
        "ajuste del regresor sobre el consumo en pesos; compara Regresión "
        "Lineal contra Árbol de Regresión con R² y MSE de prueba.",
    ))

    def _soporte(clase):
        try:
            return int(mejores[mejor]["reporte"].loc[clase, "support"])
        except Exception:
            return None

    # ---------- D2 · Figura 2: matriz de confusión ----------
    fichas.append(_ficha_curada(
        f"Figura 2. Matriz de confusión del modelo {mejor}",
        "Figura 2 del reporte completo · Unidad 2 (clasificación)",
        {
            "objeto": "Figura 2 del reporte completo",
            "tipo_grafico": "mapa de calor 2x2 de la matriz de confusión",
            "modelo": mejor,
            "clases": "Baja (0) si satisfacción 1-3; Alta (1) si satisfacción 4-5",
            "registros_en_prueba": n_prueba,
            "F1": mejor_f1,
            "Accuracy": met_mejor.get("Accuracy"),
            "Precision": met_mejor.get("Precision"),
            "Recall": met_mejor.get("Recall"),
            "AUC": mejor_auc,
            "clientes_clase_alta_en_prueba": _soporte("1"),
        },
        {
            "que_se_ve": (
                f"Mapa de calor de 2×2 en tonos azules. Las filas son la "
                f"satisfacción real (Baja 0 / Alta 1) y las columnas la "
                f"predicción del modelo {mejor}, evaluado sobre ≈{n_prueba} "
                f"registros de prueba. El número de cada celda cuenta clientes: "
                f"la diagonal principal concentra los aciertos y las dos celdas "
                f"restantes los falsos positivos y los falsos negativos."
            ),
            "tecnica": (
                f"El modelo {mejor} obtiene F1 = {_d(mejor_f1, '{:.3f}')}, "
                f"exactitud = {_d(met_mejor.get('Accuracy'), '{:.3f}')} y "
                f"AUC = {_d(mejor_auc, '{:.3f}')}. El recall de la clase alta es "
                f"{_d(met_mejor.get('Recall'), '{:.3f}')}: de cada 100 clientes "
                f"realmente satisfechos detecta alrededor de "
                f"{_d(float(met_mejor.get('Recall') or 0) * 100, '{:.0f}')}. La "
                f"precisión de "
                f"{_d(met_mejor.get('Precision'), '{:.3f}')} indica qué "
                f"proporción de las alertas de satisfacción alta resultan "
                f"correctas."
            ),
            "simple": (
                "Piense en cuatro cajones. En dos se guardan los clientes que el "
                "modelo clasificó bien y en los otros dos los que confundió: "
                "clientes contentos marcados como molestos o clientes molestos "
                "que dejó pasar como contentos. Lo importante no es solo cuántos "
                "aciertos hay, sino qué equivocación cuesta más caro en la "
                "cafetería."
            ),
            "temario": (
                "Unidad 2 — Evaluación de clasificadores: la matriz de confusión "
                "es la base de precisión, recall y F1; sin ella, la exactitud "
                "esconde el desbalance entre clases."
            ),
            "pedagogia": (
                "Entrena la lectura asimétrica del error: permite decidir si "
                "conviene priorizar recall (no perder clientes insatisfechos) o "
                "precisión (no gastar recursos en falsas alarmas), una decisión "
                "de negocio que la exactitud nunca revela."
            ),
            "errores": (
                "• Comparar modelos solo por exactitud cuando las clases están "
                "desbalanceadas. • Leer la diagonal como total de aciertos sin "
                "mirar el tamaño de cada clase. • Suponer que la matriz describe "
                "a toda la población, cuando resume el 30% reservado para "
                "prueba."
            ),
            "no_aplica": (
                "• MSE y R²: la salida es una clase, no un valor continuo. "
                "• Silhouette o método del codo: existen etiquetas reales, no se "
                "busca estructura oculta. • Q-Learning: no hay decisiones "
                f"secuenciales que optimizar. • Deep Learning: con {n_bd} registros "
                + ("tabulares no se justifica una red que la matriz 2×2 no podría explicar mejor."
                   if n_bd < 500
                   else "tabulares, una red neuronal podría aportar capacidad de representación no lineal, aunque la matriz 2×2 seguiría siendo interpretable y suficiente para este problema binario.")
            ),
        },
        "lectura de la matriz de confusión del clasificador ganador; explica "
        "falsos positivos, falsos negativos y por qué F1 no es exactitud.",
    ))

    # ---------- D3 · Figura 3: curva ROC ----------
    fichas.append(_ficha_curada(
        f"Figura 3. Curva ROC del modelo {mejor}",
        "Figura 3 del reporte completo · Unidad 2 (evaluación de clasificadores)",
        {
            "objeto": "Figura 3 del reporte completo",
            "tipo_grafico": "curva ROC con diagonal del clasificador aleatorio",
            "modelo": mejor,
            "clase_positiva": "Satisfacción Alta (4-5)",
            "registros_en_prueba": n_prueba,
            "AUC": mejor_auc,
            "F1_en_umbral_0.5": mejor_f1,
            "umbral_operativo": 0.5,
        },
        {
            "que_se_ve": (
                f"Curva naranja que asciende desde la esquina inferior izquierda "
                f"hacia la superior izquierda, con la leyenda ROC "
                f"(AUC = {_d(mejor_auc, '{:.3f}')}) y una diagonal azul punteada "
                f"que representa al clasificador aleatorio. El eje horizontal es "
                f"la tasa de falsos positivos y el vertical la tasa de "
                f"verdaderos positivos, equivalente al recall."
            ),
            "tecnica": (
                f"El AUC = {_d(mejor_auc, '{:.3f}')} sobre ≈{n_prueba} registros "
                f"de prueba estima la probabilidad de que el modelo ordene a un "
                f"cliente satisfecho por encima de uno insatisfecho elegidos al "
                f"azar. Con F1 = {_d(mejor_f1, '{:.3f}')} en el umbral 0.5, la "
                f"curva muestra que la separación entre clases es útil pero no "
                f"perfecta: la ventaja sobre la diagonal se concentra en la zona "
                f"de sensibilidad media."
            ),
            "simple": (
                "Imagine que la curva mide la habilidad del modelo para ordenar "
                "la fila de clientes: primero los contentos y después los "
                "molestos. Mientras más se acerque a la esquina superior "
                "izquierda, mejor ordena; la diagonal punteada equivale a "
                "adivinar a la suerte."
            ),
            "temario": (
                "Unidad 2 — Curvas ROC y AUC: evalúan al clasificador en todos "
                "los umbrales posibles, no solo en el corte de 0.5."
            ),
            "pedagogia": (
                "Muestra que el umbral es una decisión operativa y no un "
                "parámetro técnico: la misma curva permite un modelo conservador "
                "o uno sensible según cuánto personal de atención haya disponible "
                "en el turno."
            ),
            "errores": (
                "• Leer el AUC como porcentaje de aciertos. • Elegir el umbral "
                "0.5 sin revisar el costo de cada tipo de error. • Comparar "
                "curvas de modelos entrenados con muestras o proporciones de "
                "clase distintas."
            ),
            "no_aplica": (
                "• MSE y R²: miden error en una escala continua, no el orden de "
                "probabilidades. • Matriz de confusión fija: no representa el "
                "barrido de umbrales que hace la ROC. • K-Means: no hay etiqueta "
                "supervisada que permita calcular tasas. • Silhouette: evalúa "
                "cohesión de grupos, no discriminación entre dos clases."
            ),
        },
        "interpretación de la curva ROC y el AUC del clasificador ganador, "
        "incluyendo el efecto del umbral de decisión.",
    ))

    # ---------- D4 · Figura 4: método del codo ----------
    fichas.append(_ficha_curada(
        "Figura 4. Selección de k mediante método del codo",
        "Figura 4 del reporte completo · Unidad 3 (K-Means)",
        {
            "objeto": "Figura 4 del reporte completo",
            "tipo_grafico": "línea de inercia contra número de clústeres k",
            "rango_k_evaluado": "1 a 10",
            "k_elegido": km.get("k", 3),
            "silhouette_k_elegido": sil,
            "total_registros": n_limpio,
        },
        {
            "que_se_ve": (
                "Gráfico de líneas con marcadores en cruz azul que une el número "
                "de clústeres k, de 1 a 10, en el eje horizontal con la inercia "
                "del modelo en el vertical, sobre una rejilla tenue. La curva "
                "desciende con fuerza al inicio y se aplana a partir de k = 3 o "
                "4, formando el codo."
            ),
            "tecnica": (
                f"La inercia (suma de distancias al cuadrado a cada centroide) "
                f"siempre disminuye al aumentar k, por lo que no se elige el "
                f"mínimo sino el punto donde la ganancia se vuelve marginal. Se "
                f"trabaja con k = {km.get('k', 3)} y silhouette = "
                f"{_d(sil, '{:.4f}')} sobre {n_limpio} registros limpios: el codo "
                f"coincide con la segmentación que el reporte adopta y que el "
                f"índice de silhouette valida de forma independiente."
            ),
            "simple": (
                "Es como decidir en cuántas secciones dividir una caja de "
                "herramientas: con una se ve todo revuelto y con diez cada cosa "
                "tiene su compartimento, pero llega un punto en que añadir "
                "divisiones ya no ayuda. El codo es justo ese punto de equilibrio "
                "entre orden y complicación."
            ),
            "temario": (
                "Unidad 3 — K-Means: elección de k con el método del codo como "
                "criterio heurístico previo a la validación con silhouette."
            ),
            "pedagogia": (
                "Enseña que en el aprendizaje no supervisado no existe una "
                "respuesta única: el número de grupos se justifica con un "
                "criterio cuantitativo más una decisión de negocio, no con una "
                "fórmula cerrada."
            ),
            "errores": (
                "• Elegir el k del mínimo de inercia, que siempre sería el "
                "mayor. • Leer el codo como prueba estadística y no como "
                "heurística visual. • Ignorar si los grupos resultantes son "
                "accionables para la cafetería."
            ),
            "no_aplica": (
                "• Métricas supervisadas como F1 o AUC: no hay etiquetas que "
                "comparar. • Regresión: no se predice un valor, se agrupan "
                "registros. • División train/test clásica: no hay "
                "generalización supervisada que medir. • Q-Learning: no existe "
                "agente ni recompensa en este problema."
            ),
        },
        "elección del número de clústeres con el método del codo y su "
        "validación con silhouette.",
    ))

    # ---------- D5 · Figura 5: clústeres K-Means en PCA 2D ----------
    fichas.append(_ficha_curada(
        "Figura 5. Visualización de clústeres K-Means en PCA 2D",
        "Figura 5 del reporte completo · Unidad 3 (clustering y PCA)",
        {
            "objeto": "Figura 5 del reporte completo",
            "tipo_grafico": "dispersión de clústeres en dos componentes principales",
            "algoritmo": "K-Means",
            "k": km.get("k", 3),
            "silhouette": sil,
            "total_registros": n_limpio,
            "variables_originales": 7,
            "componentes_pca": 2,
        },
        {
            "que_se_ve": (
                "Dispersión de puntos coloreados por clúster, en escala viridis "
                "y con borde negro, proyectados con PCA: Componente Principal 1 "
                "en el eje horizontal y Componente Principal 2 en el vertical, "
                "con barra de color que identifica cada clúster y rejilla tenue "
                "de fondo."
            ),
            "tecnica": (
                f"Con k = {km.get('k', 3)} y silhouette = {_d(sil, '{:.4f}')} "
                f"sobre {n_limpio} registros, los grupos se separan en la "
                f"proyección pero se solapan parcialmente: PCA comprime las 7 "
                f"variables estandarizadas en 2 componentes solo para poder "
                f"graficar. El silhouette es la medida de calidad del "
                f"agrupamiento; el dibujo es el apoyo visual, no la evidencia."
            ),
            "simple": (
                "Es un mapa en dos dimensiones de clientes parecidos entre sí: "
                "cada color es un grupo con hábitos de consumo y horario "
                "semejantes. Que algunos puntos queden en la frontera entre dos "
                "colores significa que ese cliente no encaja perfecto en un solo "
                "grupo, algo normal en personas reales."
            ),
            "temario": (
                "Unidad 3 — Clustering y reducción de dimensionalidad: K-Means "
                "sobre el espacio completo y PCA únicamente como técnica de "
                "visualización 2D."
            ),
            "pedagogia": (
                "Distingue entre técnica de análisis y técnica de visualización: "
                "PCA no mejora el agrupamiento, solo permite verlo; confundir "
                "ambas cosas es el error conceptual típico del tema."
            ),
            "errores": (
                "• Suponer que la separación visual es la calidad real del "
                "clúster. • Interpretar los ejes PCA como variables originales, "
                "cuando son combinaciones lineales. • Nombrar los grupos por su "
                "color en lugar de por sus características de consumo."
            ),
            "no_aplica": (
                "• AUC, F1 y matriz de confusión: no hay etiqueta de referencia "
                "para medir aciertos. • MSE de regresión: no se predice un valor "
                "continuo. • PCA con todos los componentes para reconstruir: aquí "
                "se usan 2 solo para graficar, no para modelar. • Q-Learning: no "
                "hay decisiones secuenciales que optimizar."
            ),
        },
        "visualización de los clústeres de clientes en la proyección PCA y su "
        "relación con el índice de silhouette.",
    ))

    # ---------- D6 · Figura 6: dendrograma jerárquico ----------
    fichas.append(_ficha_curada(
        "Figura 6. Dendrograma del clustering jerárquico aglomerativo",
        "Figura 6 del reporte completo · Unidad 3 (clustering jerárquico)",
        {
            "objeto": "Figura 6 del reporte completo",
            "tipo_grafico": "dendrograma con ordenamiento por distancia",
            "algoritmo": "AgglomerativeClustering",
            "linkage": "ward",
            "distancia": "euclidiana sobre variables estandarizadas",
            "total_registros": n_limpio,
            "error_jerarquico": (ctx.get("res_jer", {}) or {}).get("error"),
        },
        {
            "que_se_ve": (
                "Árbol de líneas verticales y horizontales: cada hoja corresponde "
                "a un registro, identificado por su índice de cliente en el eje "
                "horizontal, y la altura de cada unión, en el eje vertical, es la "
                "distancia a la que dos grupos se fusionan. Las uniones bajas "
                "agrupan clientes muy parecidos y los saltos largos indican "
                "separación entre grupos."
            ),
            "tecnica": (
                f"Con linkage Ward y distancia euclidiana sobre las variables "
                f"estandarizadas de {n_limpio} registros, el dendrograma resume "
                f"todas las particiones posibles: cortar a una altura produce un "
                f"número concreto de grupos. Los saltos de mayor pendiente son "
                f"candidatos a corte y deben coincidir con los valores de k "
                f"razonables del método del codo."
            ),
            "simple": (
                "Es un árbol genealógico de clientes: los que se juntan muy abajo "
                "son casi gemelos en hábitos y los que se juntan muy arriba "
                "pertenecen a mundos distintos. Cortar el árbol con una línea "
                "horizontal equivale a decidir cuántos grupos se quieren "
                "distinguir."
            ),
            "temario": (
                "Unidad 3 — Clustering jerárquico aglomerativo: linkage Ward y "
                "lectura del dendrograma como alternativa a K-Means."
            ),
            "pedagogia": (
                "Aporta un criterio estructural que K-Means no ofrece: no exige "
                "fijar k de antemano, muestra la jerarquía completa y permite "
                "justificar el corte con evidencia visual."
            ),
            "errores": (
                "• Cortar el árbol en la altura mínima para obtener muchos "
                "grupos. • Confundir la altura de una unión con una probabilidad. "
                "• Comparar dendrogramas construidos con distinta métrica o "
                "distinto linkage como si fueran equivalentes."
            ),
            "no_aplica": (
                "• Método del codo: mide la inercia de K-Means, no aplica a un "
                "árbol ya construido. • F1, AUC o matriz de confusión: no hay "
                "etiquetas. • Regresión: no se estima un valor continuo. • PCA "
                "como paso obligatorio: la distancia se calcula en el espacio "
                "estandarizado completo, no en componentes."
            ),
        },
        "lectura del dendrograma y decisión de dónde cortar el árbol para "
        "obtener grupos.",
    ))

    # ---------- D7 · Figura 7: curva de aprendizaje Q-Learning ----------
    fichas.append(_ficha_curada(
        "Figura 7. Curva de aprendizaje del agente Q-Learning",
        "Figura 7 del reporte completo · Unidad 4 (aprendizaje por refuerzo)",
        {
            "objeto": "Figura 7 del reporte completo",
            "tipo_grafico": "recompensa acumulada por episodio",
            "algoritmo": "Q-Learning tabular con política epsilon-greedy",
            "episodios": episodios,
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 0.1,
            "recompensa_promedio_primeros_50": rl_ini,
            "recompensa_promedio_ultimos_50": rl_fin,
            "estados": ["Baja", "Media", "Alta"],
            "acciones": ["1 barista", "2 baristas", "3 baristas"],
        },
        {
            "que_se_ve": (
                f"Gráfico de líneas con la recompensa acumulada en el eje "
                f"vertical a lo largo de {episodios} episodios en el horizontal, "
                f"con rejilla tenue. La curva arranca con alta variabilidad, "
                f"oscila en la zona baja y se estabiliza en valores más altos "
                f"hacia el final."
            ),
            "tecnica": (
                f"La recompensa promedio pasa de {_d(rl_ini, '{:.1f}')} en los "
                f"primeros 50 episodios a {_d(rl_fin, '{:.1f}')} en los últimos "
                f"50, con α = 0.1, γ = 0.9 y ε = 0.1. Esa mejora sostenida es la "
                f"evidencia de convergencia: la política epsilon-greedy reduce la "
                f"exploración y explota la acción con mayor valor Q aprendido en "
                f"cada estado."
            ),
            "simple": (
                "Es el aprendizaje por ensayo y error de un barista nuevo: al "
                "principio prueba asignaciones al azar y a veces se equivoca; con "
                "la práctica repite lo que funcionó. La curva es su curva de "
                "experiencia: sube y se estabiliza cuando ya sabe qué hacer en "
                "cada nivel de afluencia."
            ),
            "temario": (
                "Unidad 4 — Aprendizaje por refuerzo: curva de aprendizaje como "
                "diagnóstico de convergencia de Q-Learning tabular."
            ),
            "pedagogia": (
                "Hace visible un concepto abstracto: la tensión entre explorar y "
                "explotar, y cómo el agente la resuelve solo con la señal de "
                "recompensa, sin etiquetas ni supervisión."
            ),
            "errores": (
                "• Leer la meseta como fracaso, cuando es la señal de "
                "convergencia. • Suponer que un pico alto aislado es aprendizaje "
                "estable. • Comparar curvas con distinto número de episodios o "
                "distinta función de recompensa."
            ),
            "no_aplica": (
                "• R², MSE o F1: no se predice ni se clasifica un registro, se "
                "evalúa recompensa acumulada. • División train/test supervisada: "
                "no hay etiqueta correcta que copiar. • Silhouette o método del "
                "codo: no hay agrupamiento. • Redes profundas: una tabla Q de 3 "
                "estados por 3 acciones se resuelve sin aproximación de función."
            ),
        },
        "convergencia del agente: comparación entre la recompensa inicial y "
        "final, y significado de explorar frente a explotar.",
    ))

    # ---------- D8 · Figura 8: matriz Q final ----------
    fichas.append(_ficha_curada(
        "Figura 8. Matriz Q final: valor por estado y acción",
        "Figura 8 del reporte completo · Unidad 4 (Q-Learning)",
        {
            "objeto": "Figura 8 del reporte completo",
            "tipo_grafico": "mapa de calor 3x3 de valores Q",
            "estados": ["Baja", "Media", "Alta"],
            "acciones": ["1 barista", "2 baristas", "3 baristas"],
            "episodios_entrenamiento": episodios,
            "politica_optima": politica_txt,
            "valores_q": rl.get("politica_optima", []),
        },
        {
            "que_se_ve": (
                "Mapa de calor de 3×3: los estados de afluencia (Baja, Media, "
                "Alta) en un eje y las acciones de personal (1, 2 y 3 baristas) "
                "en el otro. Cada celda muestra el valor Q aprendido y el tono "
                "más intenso marca el valor más alto de su fila, es decir, la "
                "decisión que el agente prefiere en ese estado."
            ),
            "tecnica": (
                f"La matriz Q es la tabla final de valores de acción tras "
                f"{episodios} episodios. Para cada estado, la acción con mayor Q "
                f"define la política óptima aprendida, que resulta "
                f"{politica_txt}. Los valores Q no son pesos ni pesos mexicanos: "
                f"son la recompensa futura esperada que el agente estimó para "
                f"cada par estado-acción."
            ),
            "simple": (
                "Es la chuleta que el agente terminó memorizando: para cada nivel "
                "de gente en la cafetería, cuántos empleados conviene poner. Se "
                "lee eligiendo el número más grande de cada fila; esa elección es "
                "la política del negocio."
            ),
            "temario": (
                "Unidad 4 — Ecuación de Bellman y política óptima: la matriz Q "
                "materializa el valor de cada par estado-acción."
            ),
            "pedagogia": (
                "Convierte un algoritmo en una tabla consultable: enseña que la "
                "salida del refuerzo no es un modelo predictivo, sino una regla "
                "de decisión explícita y auditable por el personal."
            ),
            "errores": (
                "• Elegir el máximo global en lugar del máximo por estado. • "
                "Interpretar los valores Q como ganancia en pesos. • Suponer que "
                "la tabla sigue siendo válida si cambian la recompensa o el "
                "patrón de demanda."
            ),
            "no_aplica": (
                "• F1, AUC o MSE: no hay predicción sobre registros ni etiqueta "
                "de referencia. • K-Means: no se agrupan clientes, se valoran "
                "acciones. • PCA: nueve celdas no requieren reducción de "
                "dimensionalidad. • Regresión logística: las probabilidades de "
                "transición son del simulador, no un modelo entrenado sobre "
                "clientes."
            ),
        },
        "lectura de la matriz Q como tabla de decisión: máximo por estado y "
        "significado del valor Q.",
    ))

    # ---------- D9 · Tabla 1: indicadores clave del proyecto ----------
    fichas.append(_ficha_curada(
        "Tabla 1. Indicadores clave del proyecto",
        "Tabla 1 del reporte completo · Resumen ejecutivo",
        {
            "objeto": "Tabla 1 del reporte completo",
            "tipo_tabla": "tabla de indicadores (dos columnas, siete filas)",
            "registros_en_bd": n_bd,
            "consumo_promedio_mxn": consumo_prom,
            "consumo_total_mxn": consumo_total,
            "satisfaccion_promedio_sobre_5": sat_prom,
            "porcentaje_hora_pico": pct_pico,
            "mejor_clasificador": mejor,
            "F1_mejor_clasificador": mejor_f1,
            "politica_afluencia_alta": politica_alta,
        },
        {
            "que_se_ve": (
                "Tabla de dos columnas y siete filas que condensa el proyecto: "
                "registros en base de datos, consumo promedio en pesos, consumo "
                "total, satisfacción promedio sobre 5, porcentaje de registros en "
                "hora pico, mejor clasificador con su F1 y la acción recomendada "
                "cuando la afluencia es alta."
            ),
            "tecnica": (
                f"Los valores del run son {n_bd} registros, consumo promedio "
                f"${_d(consumo_prom)} MXN, satisfacción "
                f"{_d(sat_prom, '{:.2f}')} de 5 y {_d(pct_pico, '{:.1f}')}% de "
                f"visitas en hora pico. El mejor clasificador es {mejor} con "
                f"F1 = {_d(mejor_f1, '{:.3f}')} y la política de refuerzo "
                f"recomienda {politica_alta} para afluencia alta."
            ),
            "simple": (
                "Es el tablero de un solo vistazo: cuántos clientes se "
                "registraron, cuánto gastan, qué tan satisfechos están y qué "
                "conviene hacer cuando hay mucha gente. Si solo se pudiera leer "
                "una tabla del reporte, sería esta."
            ),
            "temario": (
                "Resumen ejecutivo — síntesis de indicadores: integra estadística "
                "descriptiva, evaluación de clasificadores y política de refuerzo "
                "en una sola vista."
            ),
            "pedagogia": (
                "Enseña a construir un resumen ejecutivo honesto: pocos "
                "indicadores, cada uno con su unidad y su origen, en lugar de una "
                "lista larga de métricas sin jerarquía."
            ),
            "errores": (
                "• Confundir el promedio con la mediana cuando hay valores "
                "extremos. • Leer la satisfacción de 4 sobre 5 como si fuera una "
                "calificación escolar. • Tomar el F1 como medida de éxito del "
                "negocio y no como medida de calidad del clasificador."
            ),
            "no_aplica": (
                "• Intervalos de confianza: el resumen no los calcula y con la "
                "muestra actual serían muy anchos. • Pruebas de hipótesis: la "
                "tabla describe, no contrasta. • Series de tiempo: no hay columna "
                "de fecha continua, solo hora de llegada. • Deep Learning: no es "
                "una tabla de modelado."
            ),
        },
        "síntesis ejecutiva: qué indicador resume cada unidad del proyecto y "
        "cómo se leen juntos.",
    ))

    # ---------- D10 · Tabla 2: estadísticas descriptivas ----------
    fichas.append(_ficha_curada(
        "Tabla 2. Estadísticas descriptivas de variables numéricas",
        "Tabla 2 del reporte completo · Unidad 1 (estadística descriptiva)",
        {
            "objeto": "Tabla 2 del reporte completo",
            "tipo_tabla": "media y desviación estándar por variable numérica",
            "variables": ["Hora_Llegada", "Tiempo_Espera_Min",
                          "Tiempo_Servicio_Min", "Consumo_Total_MXN"],
            "estadisticas": est_desc,
            "registros_analizados": n_bd,
        },
        {
            "que_se_ve": (
                "Tabla de cuatro filas, una por variable numérica (hora de "
                "llegada, tiempo de espera, tiempo de servicio y consumo total), "
                "y dos columnas con la media y la desviación estándar calculadas "
                "sobre los registros válidos del dataset."
            ),
            "tecnica": (
                f"La hora de llegada promedia {_media('Hora_Llegada')} con "
                f"desviación {_std('Hora_Llegada')}, lo que confirma la "
                f"concentración matutina. El consumo promedio es "
                f"${_d(consumo_prom)} MXN, el tiempo de espera medio "
                f"{_d(espera_prom)} min con máximo de {_d(espera_max)} min y el "
                f"servicio promedio {_d(servicio_prom)} min. La desviación del "
                f"consumo es la mayor en términos relativos: señal de alta "
                f"dispersión entre tipos de cliente."
            ),
            "simple": (
                "Es la radiografía de un día normal: a qué hora suele llegar la "
                "gente, cuánto espera y cuánto gasta. La media dice el centro y "
                "la desviación dice cuánto varía; una desviación grande en "
                "consumo significa que conviven clientes que gastan poco con "
                "otros que gastan mucho."
            ),
            "temario": (
                "Unidad 1 — Estadística descriptiva: medidas de tendencia central "
                "y de dispersión como primer paso del análisis exploratorio."
            ),
            "pedagogia": (
                "Instala el hábito de mirar la dispersión y no solo el promedio: "
                "dos cafeterías con el mismo consumo medio pueden requerir "
                "decisiones opuestas según su variabilidad."
            ),
            "errores": (
                "• Decidir con la media ignorando la desviación estándar. • "
                "Comparar el tamaño de las desviaciones de variables con "
                "unidades distintas. • Asumir normalidad sin revisar la forma de "
                "la distribución."
            ),
            "no_aplica": (
                "• Correlación de Pearson entre variables categóricas: no están "
                "en la tabla porque su codificación no tiene sentido métrico. • "
                "Percentiles: la tabla reporta media y desviación, no cuartiles. "
                "• Inferencia poblacional: se describe la muestra disponible. • "
                "PCA: describir no requiere reducir dimensiones."
            ),
        },
        "lectura conjunta de media y desviación estándar; qué decisiones se "
        "sostienen y cuáles no con estos estadísticos.",
    ))

    # ---------- D11 · Tabla 3: outliers por IQR ----------
    fichas.append(_ficha_curada(
        "Tabla 3. Detección de outliers por método IQR",
        "Tabla 3 del reporte completo · Unidad 1 (limpieza de datos)",
        {
            "objeto": "Tabla 3 del reporte completo",
            "tipo_tabla": "cuartiles y conteo de valores atípicos por variable",
            "regla": "atípico si valor < Q1 - 1.5*IQR o valor > Q3 + 1.5*IQR",
            "registros_totales": n_bd,
            "outliers_totales": n_out,
            "porcentaje_outliers": pct_out,
            "detalle_por_variable": outliers_por_var,
        },
        {
            "que_se_ve": (
                "Tabla con una fila por variable numérica que muestra el primer "
                "cuartil Q1, el tercer cuartil Q3, el número de valores atípicos "
                "detectados y su porcentaje respecto del total de registros."
            ),
            "tecnica": (
                f"Se detectaron {n_out} valores atípicos, el {_d(pct_out)}% de "
                f"{n_bd} registros. La regla marca como atípico todo valor fuera "
                f"del intervalo entre Q1 − 1.5·IQR y Q3 + 1.5·IQR. El consumo y "
                f"el tiempo de espera aportan la mayor parte de los casos, "
                f"coherente con la cola larga típica del gasto y de la duración "
                f"del servicio."
            ),
            "simple": (
                "Sirve para encontrar los casos raros: el pedido enorme o la "
                "espera de veinte minutos. La regla de los cuartiles dice dónde "
                "termina lo normal y empieza lo excepcional. No hay que borrarlos "
                "de inmediato: primero hay que preguntar si son errores de "
                "captura o clientes reales."
            ),
            "temario": (
                "Unidad 1 — Limpieza de datos: detección de valores atípicos por "
                "rango intercuartílico (IQR) y decisión informada sobre su "
                "tratamiento."
            ),
            "pedagogia": (
                "Enseña que un outlier es una pregunta y no un error: obliga a "
                "distinguir entre error de medición, caso legítimo extremo y "
                "falla del supuesto de normalidad."
            ),
            "errores": (
                "• Eliminar todos los atípicos de forma automática. • Aplicar la "
                "regla IQR a variables categóricas codificadas. • Confundir el "
                "porcentaje de outliers con el porcentaje de datos inválidos."
            ),
            "no_aplica": (
                "• Z-score con umbral 3: supone normalidad que estas variables no "
                "cumplen. • Isolation Forest: "
                + (f"con {n_bd} registros el modelo aprende el ruido."
                   if n_bd < 500
                   else f"con {n_bd} registros podría aplicarse, aunque la regla IQR es más interpretable y suficiente para este problema.")
                + " • K-Means para detectar atípicos: fuerza a todo "
                "punto a pertenecer a un grupo. • Regresión: la tabla diagnostica, "
                "no predice."
            ),
        },
        "tratamiento de valores atípicos: qué hacer y qué no hacer con los casos "
        "detectados por IQR.",
    ))

    # ---------- D12 · Tabla 4: métricas de regresión ----------
    fichas.append(_ficha_curada(
        "Tabla 4. Métricas de evaluación de modelos de regresión",
        "Tabla 4 del reporte completo · Unidad 1 (métricas de regresión)",
        {
            "objeto": "Tabla 4 del reporte completo",
            "tipo_tabla": "MSE y R2 por modelo de regresión",
            "target": "Consumo_Total_MXN",
            "regresion_lineal_MSE": reg.get("lr_mse"),
            "regresion_lineal_R2": reg.get("lr_r2"),
            "arbol_regresion_MSE": reg.get("dt_mse"),
            "arbol_regresion_R2": reg.get("dt_r2"),
            "mejor_por_R2": mejor_reg,
            "registros_en_prueba": n_prueba,
        },
        {
            "que_se_ve": (
                "Tabla de cuatro filas que compara la Regresión Lineal y el Árbol "
                "de Regresión con dos métricas: MSE, el error cuadrático medio, y "
                "R², el coeficiente de determinación. Junto a cada R² aparece un "
                "semáforo textual que resume su calidad."
            ),
            "tecnica": (
                f"En prueba, la Regresión Lineal obtiene MSE = "
                f"{_d(reg.get('lr_mse'))} y R² = "
                f"{_d(reg.get('lr_r2'), '{:.4f}')}; el Árbol de Regresión, "
                f"MSE = {_d(reg.get('dt_mse'))} y R² = "
                f"{_d(reg.get('dt_r2'), '{:.4f}')}. El mejor por R² es "
                f"{mejor_reg}. Ninguno de los dos captura la mayor parte de la "
                f"varianza del consumo: las variables disponibles no incluyen "
                f"día de la semana, clima ni promociones, y el R² de prueba lo "
                f"refleja."
            ),
            "simple": (
                "Son dos termómetros del error: el MSE dice cuánto se equivoca el "
                "modelo en pesos al cuadrado y el R² qué parte de la variación del "
                "gasto logra explicar. Menor MSE y mayor R² indican mejor modelo, "
                "pero ninguno de los dos mide si la cafetería gana más dinero."
            ),
            "temario": (
                "Unidad 1 — Métricas de regresión: MSE y R² como criterios "
                "complementarios de evaluación."
            ),
            "pedagogia": (
                "Muestra por qué no se elige un modelo con una sola métrica: MSE "
                "y R² pueden favorecer modelos distintos y el criterio final "
                "depende del uso operativo del pronóstico."
            ),
            "errores": (
                "• Comparar MSE entre variables con unidades distintas. • Leer R² "
                "como porcentaje de aciertos. • Elegir el modelo con mayor R² en "
                "entrenamiento sin revisar la prueba."
            ),
            "no_aplica": (
                "• F1, precisión y recall: la salida es continua. • Matriz de "
                "confusión: no hay clases que cruzar. • Silhouette: no hay "
                "agrupamiento. • AUC: no hay probabilidades de clase que ordenar."
            ),
        },
        "elección entre dos regresores usando MSE y R², y advertencia sobre "
        "comparar métricas de escalas distintas.",
    ))

    # ---------- D13 · Tabla 5: comparación de clasificadores ----------
    fichas.append(_ficha_curada(
        "Tabla 5. Comparación de clasificadores: F1 y Accuracy",
        "Tabla 5 del reporte completo · Unidad 2 (métricas de clasificación)",
        {
            "objeto": "Tabla 5 del reporte completo",
            "tipo_tabla": "F1 y exactitud por clasificador",
            "clase_positiva": "Satisfacción Alta (4-5)",
            "modelos": modelos,
            "mejor_modelo": mejor,
            "mejor_F1": mejor_f1,
            "mejor_AUC": mejor_auc,
            "registros_en_prueba": n_prueba,
            "registros_entrenamiento": max(n_bd - n_prueba, 1),
        },
        {
            "que_se_ve": (
                "Tabla con una fila por clasificador —Regresión Logística, KNN, "
                "SVM y Árbol de Decisión— y una columna que combina F1 y "
                "exactitud, cada valor acompañado de un semáforo textual de "
                "desempeño."
            ),
            "tecnica": (
                f"Sobre ≈{n_prueba} registros de prueba, {mejor} obtiene el mayor "
                f"F1 = {_d(mejor_f1, '{:.3f}')} con exactitud "
                f"{_d(met_mejor.get('Accuracy'), '{:.3f}')}. La tabla pone a los "
                f"cuatro algoritmos en la misma escala: los modelos lineales y de "
                f"margen tienden a empatar en F1, mientras el Árbol concentra su "
                f"ventaja en entrenamiento y pierde más al pasar a prueba."
            ),
            "simple": (
                "Es la carrera de los cuatro candidatos: todos predicen si un "
                "cliente quedará satisfecho y la tabla dice quién acertó más. Gana "
                "el que mejor equilibra equivocarse poco con no dejar pasar "
                "clientes molestos, que es exactamente lo que resume el F1."
            ),
            "temario": (
                "Unidad 2 — Selección y comparación de modelos de clasificación: "
                "F1 como métrica principal frente a la exactitud."
            ),
            "pedagogia": (
                "Justifica la elección del modelo con evidencia y no con "
                "preferencia: muestra que el algoritmo más complejo no siempre "
                "gana y que la comparación debe hacerse sobre el mismo conjunto "
                "de prueba."
            ),
            "errores": (
                "• Elegir por exactitud cuando las clases están desbalanceadas. • "
                "Comparar modelos entrenados con particiones distintas de los "
                "datos. • Suponer que el mejor F1 implica el mejor desempeño "
                "operativo."
            ),
            "no_aplica": (
                "• MSE y R²: métricas de regresión. • Validación cruzada como "
                "criterio único: aquí se reporta la partición 70/30. • AUC como "
                "criterio de selección: se prioriza F1 por el costo asimétrico de "
                "los errores. • Deep Learning: el tamaño de muestra no lo "
                "justifica."
            ),
        },
        "comparación de los cuatro clasificadores y por qué se elige el ganador "
        "por F1 y no por exactitud.",
    ))

    # ---------- D14 · Tabla 6: comparación train vs test ----------
    deltas = []
    try:
        for nombre_m, k_test, k_train in (
            ("Regresión Lineal", "metricas_lr", "metricas_lr_train"),
            ("Árbol de Regresión", "metricas_dt", "metricas_dt_train"),
        ):
            r2_tr = float(res_reg[k_train]["R2"])
            r2_te = float(res_reg[k_test]["R2"])
            deltas.append({"modelo": nombre_m, "metrica": "R2",
                           "train": round(r2_tr, 3), "test": round(r2_te, 3),
                           "delta": round(r2_tr - r2_te, 3)})
    except (KeyError, TypeError, ValueError):
        pass
    for nombre_m, r in mejores.items():
        try:
            f1_tr = float(r["metricas_train"]["F1"])
            f1_te = float(r["metricas"]["F1"])
            deltas.append({"modelo": nombre_m, "metrica": "F1",
                           "train": round(f1_tr, 3), "test": round(f1_te, 3),
                           "delta": round(f1_tr - f1_te, 3)})
        except (KeyError, TypeError, ValueError):
            continue
    con_overfitting = [d["modelo"] for d in deltas if d["delta"] > 0.15]
    overfitting_txt = ", ".join(con_overfitting) or "ningún modelo"

    fichas.append(_ficha_curada(
        "Tabla 6. Comparación train vs test para detección de overfitting",
        "Tabla 6 del reporte completo · Unidad 2 (generalización)",
        {
            "objeto": "Tabla 6 del reporte completo",
            "tipo_tabla": "métrica en train, en test y diferencia por modelo",
            "umbral_overfitting": 0.15,
            "modelos_con_delta_mayor_a_0.15": overfitting_txt,
            "detalle": deltas,
            "registros_entrenamiento": max(n_bd - n_prueba, 1),
            "registros_prueba": n_prueba,
        },
        {
            "que_se_ve": (
                "Tabla con una fila por modelo y una columna que muestra el valor "
                "de la métrica en entrenamiento, su valor en prueba y la "
                "diferencia entre ambos, acompañada de la etiqueta OVERFITTING o "
                "OK según si la brecha supera 0.15."
            ),
            "tecnica": (
                f"Los regresores se evalúan con R² y los clasificadores con F1, "
                f"sobre {max(n_bd - n_prueba, 1)} registros de entrenamiento y "
                f"{n_prueba} de prueba. Una diferencia cercana a cero indica "
                f"generalización; por encima del umbral de 0.15 hay memorización. "
                f"En este run superan el umbral: {overfitting_txt}. El modelo a "
                f"desplegar es el de menor brecha con F1 competitivo, no el de "
                f"mejor métrica en entrenamiento."
            ),
            "simple": (
                "Es como comparar los resultados del examen de práctica con el "
                "examen real: si al estudiante le va perfecto en la práctica y "
                "mal en el real, memorizó las respuestas. Aquí se busca al modelo "
                "que mantiene su calificación cuando ve clientes que nunca había "
                "visto."
            ),
            "temario": (
                "Unidad 2 — Generalización y overfitting: comparación train vs "
                "test y regla práctica del delta mayor a 0.15."
            ),
            "pedagogia": (
                "Enseña a desconfiar del desempeño perfecto: la distancia entre "
                "entrenamiento y prueba es la señal de alarma más útil y la más "
                "ignorada al reportar resultados."
            ),
            "errores": (
                "• Reportar solo el desempeño en entrenamiento. • Elegir el modelo "
                "con mayor F1 sin mirar la brecha. • Confundir overfitting con un "
                "modelo mal entrenado (underfitting)."
            ),
            "no_aplica": (
                "• Curva ROC: mide discriminación en un umbral, no la brecha "
                "train/test. • Método del codo y silhouette: propios del no "
                "supervisado. • Q-Learning: su diagnóstico es la curva de "
                "recompensa. • Pruebas de significancia estadística: la tabla es "
                "descriptiva."
            ),
        },
        "diagnóstico de overfitting con la brecha train/test y su efecto en la "
        "decisión de despliegue.",
    ))

    # ---------- D15 · Tabla 7: cobertura de la rúbrica ----------
    rubrica = [
        "1. Definición del problema", "2. Objetivos", "3. Marco conceptual",
        "4. Análisis de datos", "5. Preprocesamiento", "6. División de datos",
        "7. Selección del modelo", "8. Entrenamiento", "9. Evaluación",
        "10. Implementación", "11. Resultados y conclusiones", "12. Entregables",
    ]
    fichas.append(_ficha_curada(
        "Tabla 7. Cobertura de los 12 puntos de la rúbrica",
        "Tabla 7 del reporte completo · Cierre y trazabilidad",
        {
            "objeto": "Tabla 7 del reporte completo",
            "tipo_tabla": "trazabilidad de 12 requisitos contra capítulos y app",
            "puntos_de_la_rubrica": rubrica,
            "total_puntos": len(rubrica),
            "capitulos_del_reporte": 15,
            "mejor_clasificador": mejor,
        },
        {
            "que_se_ve": (
                "Tabla de doce filas, una por cada punto de la rúbrica del "
                "Proyecto Final —definición del problema, objetivos, marco "
                "conceptual, análisis de datos, preprocesamiento, división de "
                "datos, selección del modelo, entrenamiento, evaluación, "
                "implementación, resultados y entregables—, con la referencia al "
                "capítulo o a la funcionalidad de la app donde se cubre cada uno."
            ),
            "tecnica": (
                "La tabla es trazabilidad, no una métrica de desempeño: cada "
                "punto se mapea a un capítulo numerado del reporte o a una función "
                "concreta de la aplicación Streamlit. Los doce puntos quedan "
                "cubiertos con evidencia verificable, incluida la app desplegada "
                "en Streamlit Community Cloud, el repositorio versionado y la "
                "base de datos en Supabase."
            ),
            "simple": (
                "Es la lista de cotejo del profesor: doce requisitos y, al lado de "
                "cada uno, dónde está resuelto. Sirve para revisar el proyecto sin "
                "leer el reporte completo y para comprobar que no falta nada."
            ),
            "temario": (
                "Cierre y trazabilidad — alineación con la rúbrica: cada técnica "
                "del temario se evidencia en un capítulo y en una funcionalidad de "
                "la aplicación."
            ),
            "pedagogia": (
                "Enseña a documentar con trazabilidad: un entregable profesional "
                "demuestra dónde está cada requisito cumplido en lugar de afirmar "
                "que se cumplió."
            ),
            "errores": (
                "• Confundir esta tabla con una evaluación de calidad del "
                "proyecto. • Marcar cobertura sin una referencia verificable. • "
                "Suponer que cubrir los doce puntos implica un buen modelo."
            ),
            "no_aplica": (
                "• Métricas de desempeño: es una tabla de auditoría, no de "
                "resultados. • Comparaciones estadísticas: no hay variables que "
                "contrastar. • Cualquier algoritmo de ML: la tabla documenta, no "
                "modela. • Visualizaciones: su valor está en la referencia, no en "
                "un gráfico."
            ),
        },
        "cobertura de la rúbrica como trazabilidad: dónde se evidencia cada "
        "requisito dentro del reporte y de la app.",
    ))

    return fichas


def _campos_de_ficha(spec: dict) -> tuple:
    """Genera los siete bloques de una ficha y reporta si usó la IA.

    Devuelve `(campos, fuente)`. Si la IA no está disponible, falla o responde
    un JSON incompleto, `analizar_seccion_ia` ya devuelve el respaldo curado
    bloque por bloque, de modo que esta función nunca lanza.
    """
    fallback = spec["fallback"]
    try:
        ia = analizar_seccion_ia(
            f"Ficha didáctica · {spec['titulo']}",
            spec["datos"],
            fallback.get("tecnica", ""),
            instruccion_extra=spec["instruccion"],
            doble_capa=False,      # el prompt didáctico ya pide el bloque simple
            tipo="didactico",
            fallback_campos=fallback,
        )
    except Exception:
        return dict(fallback), "respaldo"
    if not isinstance(ia, dict) or not ia:
        return dict(fallback), "respaldo"
    return ia, ("ia" if ia != fallback else "respaldo")


def _fichas_didacticas(ctx: dict, df=None) -> list[dict]:
    """Construye las 15 fichas del reporte didáctico con IA en paralelo.

    Sustituye a `_fichas_didacticas_placeholder` de la Fase 5A: la estructura
    de cada ficha (`titulo`, `referencia`, `campos`) no cambia, solo el origen
    de los textos. Cada ficha envía a Gemini su propio payload acotado (las
    métricas reales del objeto que explica) y su respaldo curado, así que un
    fallo puntual no contamina a las demás. Con `_FICHA_WORKERS = 4` las 15
    llamadas tardan lo de unas cuatro, y la caché de `mod_ia_analista` evita
    repetir cuota si se regenera el reporte con los mismos datos.
    """
    especificaciones = _especificaciones_didacticas(ctx, df)
    campos = [dict(spec["fallback"]) for spec in especificaciones]
    fuentes = ["respaldo"] * len(especificaciones)

    with ThreadPoolExecutor(max_workers=_FICHA_WORKERS) as executor:
        futuros = {
            executor.submit(_campos_de_ficha, spec): i
            for i, spec in enumerate(especificaciones)
        }
        for fut in as_completed(futuros):
            i = futuros[fut]
            try:
                resultado, fuente = fut.result()
            except Exception:
                continue
            if isinstance(resultado, dict) and resultado:
                campos[i] = resultado
                fuentes[i] = fuente

    return [
        {
            "titulo": spec["titulo"],
            "referencia": spec["referencia"],
            "campos": campos[i],
            "fuente": fuentes[i],
        }
        for i, spec in enumerate(especificaciones)
    ]


def _cuerpo_didactico(pdf, fichas: list, df=None):
    """Dibuja la introducción y las fichas; registra cada ficha en el índice."""
    pdf.add_page()
    pdf.cap_actual = ""
    pdf.set_x(pdf.l_margin)
    pdf.set_font(pdf.FONT, "B", 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "Reporte Didáctico", 0, 1, "L")
    pdf.set_font(pdf.FONT, "", 9)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 5, (
        "Este documento acompaña al reporte técnico y explica, una por una, "
        "cada figura y cada tabla que aparecen en él. La idea es que cualquier "
        "persona pueda reconocer el concepto de Aprendizaje Automático que hay "
        "detrás de cada resultado, sin necesidad de dominar la notación "
        "matemática de antemano."
    ))
    pdf.ln(2)
    pdf.subtitulo("Cómo leer cada ficha")
    pdf.parrafo(
        "Cada ficha sigue siempre el mismo orden de siete bloques. Los seis "
        "primeros construyen la comprensión: qué se ve, qué significa "
        "técnicamente, cómo se explicaría en palabras simples, a qué unidad del "
        "temario corresponde, por qué importa aprenderlo y qué errores se "
        "cometen al interpretarlo. El séptimo bloque responde una pregunta que "
        "casi nunca se plantea: qué técnicas del temario NO se aplican en esa "
        "figura o tabla y por qué. Ese bloque evita el error de creer que toda "
        "técnica sirve para cualquier problema."
    )
    pdf.subtitulo("Estructura de cada ficha")
    pdf.tabla_larga(
        [(etiqueta, _DESCRIPCION_CAMPO.get(clave, ""))
         for clave, etiqueta in CAMPOS_DIDACTICOS],
        ancho1=62, ancho2=128,
    )
    pdf.subtitulo("Mapa de fichas incluidas")
    pdf.tabla_larga(
        [(f"D{i}. {f['titulo']}", f["referencia"])
         for i, f in enumerate(fichas, 1)],
        ancho1=75, ancho2=115,
    )
    pdf.set_font(pdf.FONT, "I", 8)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 4, (
        "Versión 5.1 — contenido generado con IA a partir de las métricas "
        "reales de este proyecto, con un respaldo redactado sobre las mismas "
        "cifras cuando el servicio no está disponible."
    ))
    pdf.set_text_color(0, 0, 0)

    # --- Fichas: una por página, registradas en el índice ---
    pdf.cap_actual = "Reporte Didáctico · Fichas"
    for i, ficha in enumerate(fichas, 1):
        pdf.add_page()
        pdf.indice_caps[f"D{i}. {ficha['titulo']}"] = pdf.page_no()
        pdf.ficha_didactica(
            i,
            ficha["titulo"],
            ficha.get("campos") or {},
            referencia=ficha.get("referencia"),
        )


def _render_didactico(fichas: list, df=None) -> tuple:
    """Doble pasada del reporte didáctico: borrador, medición y final.

    Render #1 (borrador) dibuja portada + introducción + fichas y registra la
    página real de cada ficha en `indice_caps`.
    La medición dibuja portada + el índice de fichas en un PDF desechable para
    saber cuántas páginas ocupa (no se asume un offset fijo: con 15 fichas el
    índice puede pasar de una página).
    Render #2 (final) antepone el índice con las páginas desplazadas por ese
    offset, de modo que la numeración coincide con el documento real.

    `total_pages` se fija antes de dibujar para que el pie imprima
    "Página N de T" sin una tercera pasada.
    """
    # ---------- RENDER #1: BORRADOR ----------
    pdf1 = ReporteCafeteria()
    pdf1.modo = "borrador"
    pdf1.set_auto_page_break(auto=True, margin=15)
    pdf1.portada_institucional()
    _cuerpo_didactico(pdf1, fichas, df)
    paginas_borrador = pdf1.page_no()

    # ---------- MEDICIÓN: páginas que ocupa el índice ----------
    pdf_med = ReporteCafeteria()
    pdf_med.set_auto_page_break(auto=True, margin=15)
    pdf_med.portada_institucional()
    pdf_med._cabecera_indice("Índice de Fichas Didácticas")
    for titulo in pdf1.indice_caps:
        pdf_med._fila_indice(titulo, 10)
    pdf_med._saltar_header_footer(False)
    offset = pdf_med.page_no() - 1          # -1 = portada
    paginas_total = paginas_borrador + offset

    # ---------- RENDER #2: FINAL ----------
    pdf2 = ReporteCafeteria()
    pdf2.modo = "final"
    pdf2.total_pages = paginas_total
    pdf2.set_auto_page_break(auto=True, margin=15)
    pdf2.portada_institucional()
    pdf2._cabecera_indice("Índice de Fichas Didácticas")
    for titulo, pagina in pdf1.indice_caps.items():
        pdf2._fila_indice(titulo, pagina + offset)
    pdf2._saltar_header_footer(False)
    _cuerpo_didactico(pdf2, fichas, df)

    info = {
        "fichas": len(fichas),
        "paginas_borrador": paginas_borrador,
        "paginas_indice": offset,
        "paginas_total": paginas_total,
        "paginas_reales": pdf2.page_no(),
    }
    return pdf2, info


def generar_reporte_didactico(df: pd.DataFrame) -> tuple:
    """Genera el Reporte Didáctico (v5.1) con el contenido de la Fase 5B.

    Reutiliza `_construir_contexto` y pide a `_fichas_didacticas()` las 15
    fichas: cada una se redacta con `analizar_seccion_ia(tipo="didactico")` en
    paralelo y cae a un respaldo curado si la IA no está disponible.
    """
    if df.empty or len(df) < 20:
        return None, ("Se necesitan al menos 20 registros para generar "
                      "el reporte didáctico.")

    try:
        ctx = _construir_contexto(df)
        fichas = _fichas_didacticas(ctx, df)
        pdf, _info = _render_didactico(fichas, df)
        return bytes(pdf.output()), None

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error al generar reporte didáctico: {e}"
