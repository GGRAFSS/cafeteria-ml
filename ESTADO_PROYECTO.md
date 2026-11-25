# ESTADO DEL PROYECTO — Cafetería Universitaria ML

**Última actualización:** 2026-11-25
**Versión:** v5.1 (post Fase 5B + Nivel 5 + Matriz con iconos + Ingesta libre + Fix paginación)
**Repos:**
- Oficial: https://github.com/GGRAFSS/cafeteria-ml (`a6a8a7f`)
- App-main: https://github.com/GGRAFSS/mi-proyecto-pruebas (`271d79c`)

**App en vivo:** https://cafeteria-ml-aprendizaje-auto.streamlit.app

---

## ✅ QUÉ FUNCIONA HOY (verificado)

### Reportes PDF (3 tipos)
| Reporte | Genera | Tamaño | Páginas | Comando |
|---|---|---|---|---|
| Reporte completo | ✅ | ~500 KB | 27-28 | `mrp.generar_reporte_pdf(df)` |
| Resumen 1 página | ✅ | ~36 KB | 1-2 | `mrp.generar_resumen_pdf(df)` |
| Reporte didáctico | ✅ | ~237 KB | 19 | `mrp.generar_reporte_didactico(df)` |

### Ingesta de datos (libre, sin límite)
- ✅ Formulario manual (1 registro a la vez)
- ✅ Subir CSV/Excel (hasta 10,000 filas por archivo)
- ✅ Generador sintético (slider 1-5000, seed=42, reproducible)
- ✅ **Nuevo:** la ingesta NO tiene límite de 200. Puedes tener 1000, 2000, 5000+.
- ✅ **Fix paginación:** `load_data()` ahora pagina con `.range()` y trae el N real (no capado a 1000 por PostgREST).
- ✅ **Fix IDs:** `_max_id_db()` consulta el ID máximo real (1 fila, inmune al cap) → no colisiona al insertar >1000.

### Normalizador de BD (único límite)
- ✅ Slider 1-200, deja la BD en exactamente N registros.
- ✅ Si hay más de N: borra los más recientes (conserva los más antiguos).
- ✅ Si hay menos de N: rellena con sintéticos (seed=42).
- ✅ Nunca deja la BD vacía (mínimo 1 registro).

### Reporte adaptable al N real (Nivel 5)
El reporte **siempre lee `len(df)` de Supabase** y se adapta:
- Cap. 3: justificación de exclusiones con N real + nivel.
- Cap. 6: tabla "Tamaño mínimo idóneo" + fila "Tu dataset actual".
- Cap. 7: regla de selección + tabla "Matriz de aplicabilidad".
- Cap. 11.4: limitaciones con 5 tramos.
- Fallbacks: `ana_marco`, `ana_division`, `ana_seleccion`, `ana_validacion`.
- Fichas D2 (matriz de confusión) y D11 (outliers) con `n_bd`.

### Niveles del semáforo (helper `_nivel_muestra(n)`)
| N | Nivel | Significado |
|---|---|---|
| < 100 | ROJO | INSUFICIENTE para generalizar |
| 100-199 | AMARILLO | PRELIMINAR, ampliar recomendado |
| 200-499 | VERDE | SUFICIENTE para conclusiones estables |
| 500-999 | VERDE+ | EXCELENTE, permite ensembles |
| ≥ 1000 | VERDE++ | EXCELENTE, permite ensembles y Deep Learning tabular |

### Matriz de aplicabilidad (Cap. 7)
Tabla con 7 filas que dice qué técnicas aplican con el N real:
- Filas 1-4: técnicas clásicas (supervisado, no supervisado, refuerzo, validación cruzada) → **siempre aplican**.
- Fila 5: Ensembles (RF/XGBoost) → "⚠ Viable desde n≥500 — no aplicado (N insuficiente / se prioriza interpretabilidad)".
- Fila 6: Deep Learning tabular → "⚠ Viable desde n≥1000 — no aplicado (N insuficiente / se prioriza interpretabilidad)".
- Fila 7: Transfer Learning y GANs → "✗ No aplica: sin datos no estructurados".
- Nota al pie: explica qué es "N" y qué significan ⚠ y ✗.

**Glifos usados:** `⚠` (U+26A0) y `✗` (U+2717). Ambos renderizan en DejaVuSans.ttf. **`❌` (U+274C) NO existe en la fuente, no usar.**

---

## 🗂️ ARQUITECTURA DE ARCHIVOS

| Archivo | Rol | ¿Tocar? |
|---|---|---|
| `app.py` | UI Streamlit (6 pestañas + normalizador) | Solo cambios quirúrgicos con backup |
| `data_manager.py` | Conexión Supabase, CRUD, sintéticos, normalizador, paginación | Solo cambios quirúrgicos con backup |
| `mod_reporte.py` | Reporte completo + resumen + didáctico | Solo cambios quirúrgicos con backup |
| `mod_ia_analista.py` | Analista IA (Gemini + caché) | Solo cambios quirúrgicos con backup |
| `mod_supervisado.py` | Regresión + Clasificación | Solo `R2_train`/`F1_train` si se pide |
| `mod_no_supervisado.py` | Distancias + Clustering | No tocar sin aprobación |
| `mod_refuerzo.py` | Q-Learning | No tocar sin aprobación |
| `mod_u1_fundamentos.py` | Teoría U1 | No tocar sin aprobación |
| `stats_utils.py` | Estadísticas + outliers IQR | No tocar sin aprobación |
| `viz_utils.py` | Gráficas del temario | No tocar sin aprobación |
| `PROYECTO_CONTEXTO.md` | Arquitectura general (v4.0) | Histórico, no actualizar |
| `ESTADO_PROYECTO.md` | **Este archivo** (v5.1) | Actualizar al cerrar cada fase |

---

## 🔒 REGLAS DE TRABAJO (para Cline o cualquier IA)

1. **NUNCA** usar `PowerShell Set-Content`/`Out-File` (corrompen acentos).
2. **NUNCA** usar `-replace` de PowerShell.
3. **SÍ** usar Python (`open` con `encoding="utf-8"`, `newline=""`) para leer/escribir.
4. **SÍ** usar `str_replace` quirúrgico (no reescribir archivos completos).
5. **SÍ** backup previo: `Copy-Item archivo.py archivo_pre_<fase>.bak_$(Get-Date -Format yyyyMMddHHmm)`.
6. **SÍ** `py_compile` después de cada cambio.
7. **SÍ** verificar acentos: `"Ã"` no debe aparecer; `"cafetería"`, `"satisfacción"` SÍ.
8. **SÍ** verificar CRLF en `mod_reporte.py` (CRLF == LF count).
9. **NO** hacer commit sin aprobación explícita del usuario.
10. **NO** tocar `mod_*.py` sin justificación.

---

## 🧪 COMANDOS DE VERIFICACIÓN

### Sintaxis
```powershell
python -c "import ast; ast.parse(open('archivo.py', encoding='utf-8').read()); print('OK')"
python -m py_compile archivo.py