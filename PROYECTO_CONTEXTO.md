# Contexto del Proyecto — Cafetería Universitaria ML

## 🎯 Objetivo
Sistema de Aprendizaje Automático desplegado en Streamlit Cloud que analiza datos de una cafetería universitaria (horario 7:00-11:00 AM). Cubre las 4 unidades del temario de Maestría en IA del TecNM.

## 🏗️ Stack
- **Frontend/Backend:** Streamlit (Python 3.12)
- **Base de datos:** Supabase (PostgreSQL en la nube)
- **ML:** scikit-learn, pandas, numpy, scipy
- **PDF:** fpdf2 (con DejaVu Sans para Unicode)
- **IA generativa:** Google Gemini 2.5 Flash (vía `google-genai`)
- **Gráficas:** matplotlib, seaborn
- **Deploy:** Streamlit Community Cloud

## 📁 Archivos clave
| Archivo | Rol |
|---|---|
| `app.py` | UI Streamlit (6 pestañas) |
| `data_manager.py` | Conexión Supabase, CRUD, generador sintético |
| `mod_u1_fundamentos.py` | Unidad 1: teoría + clasificación de variables |
| `mod_supervisado.py` | Unidad 2: regresión + clasificación |
| `mod_no_supervisado.py` | Unidad 3: distancias + clustering |
| `mod_refuerzo.py` | Unidad 4: Q-Learning |
| `mod_ia_analista.py` | Analista IA con Gemini + caché |
| `mod_reporte.py` | Generador de reportes PDF |
| `stats_utils.py` | Estadísticas descriptivas + outliers IQR |
| `viz_utils.py` | Gráficas obligatorias del temario |
| `assets/logo_tecnm_virtual.png` | Logo TecNM Virtual (320×380, fondo transparente) |
| `assets/logo_maestria_ia.png` | Logo Maestría IA (274×344, fondo transparente) |

## 📋 Estado actual (v4.0) — CERRADO
- ✅ App Streamlit funcional con 6 pestañas + KPIs en sidebar
- ✅ Reporte PDF tipo tesis: 15 capítulos + front matter (portada,
  índices de contenido/figuras/tablas, Resumen Ejecutivo)
- ✅ 8 figuras + 7 tablas con captions e índices automáticos
- ✅ 27 tareas Gemini (16 análisis doble capa + 11 decisiones operativas)
- ✅ Caché IA con paralelización (ThreadPoolExecutor, max_workers=4)
- ✅ Nombres institucionales correctos: Rodríguez Avilez Selene,
  Raygoza Flores Gabriel, Dra. Claudia Ibeth González Berrelleza
- ✅ Detección de overfitting con R2_train vs R2_test y F1_train vs F1_test
- ✅ Cobertura 12/12 de la rúbrica del Proyecto Final

## 🎯 Próximos pasos
- Commit de app.py + mod_reporte.py + assets/ (logos)
- Deploy a Streamlit Cloud
- Grabar video demo (8-12 min)
- Preparar presentación PowerPoint

## ✅ Cobertura de la rúbrica (12/12)
Reporte PDF tipo tesis que cubre los 12 puntos del Proyecto Final:
1. Definición del problema ✅ (cap. 1)
2. Planteamiento de objetivos ✅ (cap. 2: general + específicos + temario)
3. Marco conceptual ✅ (cap. 3: definición de Mitchell + aplicabilidad)
4. Adquisición y análisis de datos ✅ (cap. 4 + app)
5. Preprocesamiento ✅ (cap. 5: pipeline justificado)
6. División de datos ✅ (cap. 6: 70/30 + tamaños mínimos)
7. Selección del modelo ✅ (cap. 7: qué aplica y qué no, con razones)
8. Entrenamiento del modelo ✅ (caps. 8-10: fichas técnicas e hiperparámetros)
9. Evaluación del modelo ✅ (caps. 8-10: MSE, R², F1, AUC, Silhouette)
10. Validación y análisis ✅ (cap. 11: overfitting/underfitting + mitigaciones)
11. Implementación funcional ✅ (cap. 12 + app Streamlit)
12. Entregables ✅ (cap. 13: presentación, video, repo + tabla de rúbrica)

## 🎨 Estructura del PDF (implementada)
- Portada con logos institucionales
- Índice de contenido con números de página reales (doble pasada)
- Índice de figuras y tablas
- 15 capítulos (los 12 de la rúbrica + Glosario + Referencias APA + Entregables)
- Numeración arábiga simple (no romana)
- Encabezado y pie ("Página N de T") en todas las páginas salvo portada

## ⚠️ Reglas de trabajo
1. **NO reescribir archivos completos de golpe** — proponer cambios quirúrgicos con `str_replace`
2. **Verificar sintaxis** después de cada cambio: `python -c "import ast; ast.parse(open('archivo.py',encoding='utf-8').read())"`
3. **Correr tests** después de cada fase (ver "Comandos de verificación")
4. **`app.py`**: solo cambios quirúrgicos con backup previo (`app_pre_*.bak_*`); `data_manager.py` no se tocó en v4.0
5. **NO tocar los modelos ML** salvo `mod_supervisado.py` (para R2_train/F1_train)
6. **Backup antes de cambios grandes**: `Copy-Item mod_reporte.py mod_reporte.py.bak_$(Get-Date -Format yyyyMMdd)`
7. **Fallbacks robustos**: si los logos faltan, el PDF se genera sin ellos
8. **Sin dependencias nuevas** — usar lo que ya está en `requirements.txt`

## 🧪 Comandos de verificación

### Sintaxis
```powershell
python -c "import ast; ast.parse(open('mod_reporte.py', encoding='utf-8').read()); print('Sintaxis OK')"
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('Sintaxis OK')"
python -m py_compile mod_reporte.py app.py; echo "Compile OK"
```

### Reporte PDF de prueba (datos sintéticos)
```powershell
python -c "import data_manager as dm, mod_reporte as rp; b,e=rp.generar_reporte_pdf(dm.generar_datos_sinteticos(120)); print(len(b), 'bytes', e)"
```
Resultado esperado en v4.0: ~511 KB, 26 páginas, sin errores.

### App local
```powershell
streamlit run app.py
```
En el sidebar deben verse 4 KPIs (Consumo promedio, Satisfacción promedio,
% Hora pico, Outliers detectados) con tooltips.

## 📦 Backups de esta sesión
- mod_reporte_pre_v4.bak_20261124 (antes de Fase 2)
- mod_reporte_pre_fase3.bak_20261124
- mod_reporte_pre_fase3b.bak_20261124
- mod_reporte_pre_fase3c.bak_20261124
- mod_supervisado.py.bak_20261124
- app.py.bak_20261124
- app_pre_fase3b.bak_20261124
- app_pre_fase3c.bak_20261124
- app_pre_fase4.bak_20261124

Estos backups NO se commitean: desde Fase 4 `.gitignore` incluye
`*.bak_*`, `_t_*` y `*.rar`, por lo que ya no aparecen en `git status`.
Se pueden borrar manualmente una vez validado el deploy.