"""
Módulo de Generación de Reporte PDF
Compila todos los resultados de las 4 unidades en un documento estructurado.
"""
import os
import re
import tempfile
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

import mod_supervisado as ms
import mod_no_supervisado as mns
import mod_refuerzo as mr


def _limpiar_texto(texto) -> str:
    """Reemplaza viñetas '•' por '-' y elimina caracteres fuera del rango de Helvetica."""
    if texto is None:
        return ""
    s = str(texto).replace("•", "-")
    return re.sub(r"[^\x20-\x7E\xA0-\xFF\n\r\t]", "", s)


class ReporteCafeteria(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 5, _limpiar_texto('Reporte de Aprendizaje Automático - Cafetería Universitaria'), 0, 1, 'C')
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, _limpiar_texto(f'Página {self.page_no()}'), 0, 0, 'C')

    def titulo_capitulo(self, num, titulo):
        self.add_page()
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, _limpiar_texto(f'{num}. {titulo}'), 0, 1, 'L')
        self.set_draw_color(0, 51, 102)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(5)

    def subtitulo(self, titulo):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(51, 51, 51)
        self.cell(0, 8, _limpiar_texto(titulo), 0, 1, 'L')
        self.ln(2)

    def parrafo(self, texto):
        self.set_font('Helvetica', '', 9)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, _limpiar_texto(texto))
        self.ln(3)

    def nota(self, texto):
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 4, _limpiar_texto(f'Análisis: {texto}'))
        self.ln(3)

    def tabla_metrica(self, filas):
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        self.cell(60, 7, _limpiar_texto('Métrica'), 1, 0, 'C', True)
        self.cell(60, 7, _limpiar_texto('Valor'), 1, 1, 'C', True)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(0, 0, 0)
        for k, v in filas:
            self.cell(60, 6, _limpiar_texto(k), 1, 0, 'L')
            self.cell(60, 6, _limpiar_texto(v), 1, 1, 'L')
        self.ln(3)

    def agregar_figura(self, fig, titulo=""):
        if fig is None:
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            fig.savefig(tmp.name, dpi=90, bbox_inches='tight')
            if titulo:
                self.set_font('Helvetica', 'I', 8)
                self.set_text_color(80, 80, 80)
                self.cell(0, 5, _limpiar_texto(titulo), 0, 1, 'C')
            self.image(tmp.name, x=30, w=150)
            os.unlink(tmp.name)
        plt.close(fig)
        self.ln(3)


def generar_reporte_pdf(df: pd.DataFrame) -> tuple[bytes | None, str | None]:
    """Genera un reporte PDF completo con todos los resultados de las 4 unidades."""
    if df.empty or len(df) < 20:
        return None, "Se necesitan al menos 20 registros para generar el reporte."

    try:
        pdf = ReporteCafeteria()
        pdf.set_auto_page_break(auto=True, margin=15)

        # ============ PORTADA ============
        pdf.add_page()
        pdf.ln(35)
        pdf.set_font('Helvetica', 'B', 22)
        pdf.set_text_color(0, 51, 102)
        pdf.multi_cell(0, 10, _limpiar_texto('Reporte de Aprendizaje Automático\nCafetería Universitaria'), 0, 'C')
        pdf.ln(10)
        pdf.set_font('Helvetica', '', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, _limpiar_texto('Proyecto Integrador'), 0, 1, 'C')
        pdf.ln(15)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 6, _limpiar_texto('Autores:'), 0, 1, 'C')
        pdf.cell(0, 6, _limpiar_texto('Rodríguez Avilés Selene'), 0, 1, 'C')
        pdf.cell(0, 6, _limpiar_texto('Raygoza Flores Gabriel'), 0, 1, 'C')
        pdf.ln(10)
        pdf.cell(0, 6, _limpiar_texto('Materia: Aprendizaje Automático'), 0, 1, 'C')
        pdf.cell(0, 6, _limpiar_texto('Docente: Dr. Valdez Acosta Fevrier Adolfo'), 0, 1, 'C')
        pdf.cell(0, 6, _limpiar_texto('Maestría en Inteligencia Artificial'), 0, 1, 'C')
        pdf.cell(0, 6, _limpiar_texto('Instituto Tecnológico de Tijuana - TecNM'), 0, 1, 'C')
        pdf.ln(10)
        pdf.cell(0, 6, _limpiar_texto(f'Fecha: {datetime.now().strftime("%d/%m/%Y")}'), 0, 1, 'C')
        pdf.ln(15)
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 5, _limpiar_texto('App: https://cafeteria-ml-aprendizaje-auto.streamlit.app'), 0, 1, 'C')
        pdf.cell(0, 5, _limpiar_texto('Repo: https://github.com/GGRAFSS/cafeteria-ml'), 0, 1, 'C')

        # ============ 1. INTRODUCCIÓN ============
        pdf.titulo_capitulo('1', 'Introducción y Definición del Problema')
        pdf.parrafo(
            "La cafetería universitaria presenta alta afluencia en horario matutino (7:00-11:00 AM), "
            "lo que genera cuellos de botella, insatisfacción del cliente y pérdida de ventas. "
            "La aplicación de técnicas de Aprendizaje Automático permite predecir el comportamiento, "
            "segmentar los perfiles y optimizar la asignación de personal."
        )
        pdf.subtitulo('Tipos de problemas abordados')
        pdf.parrafo(
            "- Regresión: Predicción del Consumo Total (MXN).\n"
            "- Clasificación: Predicción de Satisfacción Alta (4-5) vs Baja (1-3).\n"
            "- Agrupamiento: Segmentación con K-Means, Jerárquico y DBSCAN.\n"
            "- Refuerzo: Optimización de personal mediante Q-Learning."
        )

        # ============ 2. ANÁLISIS DE DATOS ============
        pdf.titulo_capitulo('2', 'Adquisición y Análisis de Datos')
        pdf.parrafo(
            f"El dataset contiene {len(df)} registros con 9 variables, "
            f"capturados en horario matutino."
        )
        pdf.subtitulo('Estadísticas Básicas')
        desc = df[["Hora_Llegada", "Tiempo_Espera_Min", "Tiempo_Servicio_Min", "Consumo_Total_MXN"]].describe()
        filas = []
        for col in desc.columns:
            filas.append((col, f"Media={desc.loc['mean', col]:.2f}, Std={desc.loc['std', col]:.2f}"))
        pdf.tabla_metrica(filas)
        pdf.nota(
            "Se observa variabilidad significativa en los tiempos y consumos, lo cual es "
            "apropiado para entrenar modelos predictivos."
        )

        # ============ 3. REGRESIÓN ============
        pdf.titulo_capitulo('3', 'Modelos de Regresión (U2)')
        pdf.parrafo("Predicción del Consumo Total (MXN) usando Regresión Lineal y Árboles de Regresión.")
        res_reg = ms.entrenar_regresion(df)
        if res_reg.get("error"):
            pdf.parrafo(f"Error en regresión: {res_reg['error']}")
        else:
            pdf.subtitulo('Métricas de Evaluación')
            pdf.tabla_metrica([
                ("Regresión Lineal - MSE", f"{res_reg['metricas_lr']['MSE']:.2f}"),
                ("Regresión Lineal - R²", f"{res_reg['metricas_lr']['R2']:.4f}"),
                ("Árbol de Regresión - MSE", f"{res_reg['metricas_dt']['MSE']:.2f}"),
                ("Árbol de Regresión - R²", f"{res_reg['metricas_dt']['R2']:.4f}"),
            ])
            pdf.agregar_figura(res_reg["fig"], "Figura 1: Regresión Lineal - Real vs Predicho")
            pdf.nota(
                f"El R² de {res_reg['metricas_lr']['R2']:.3f} indica que el modelo lineal explica "
                f"aproximadamente el {res_reg['metricas_lr']['R2']*100:.1f}% de la variabilidad del consumo. "
                f"El MSE de {res_reg['metricas_lr']['MSE']:.2f} significa que el error promedio es de "
                f"${np.sqrt(res_reg['metricas_lr']['MSE']):.2f} MXN por predicción."
            )

        # ============ 4. CLASIFICACIÓN ============
        pdf.titulo_capitulo('4', 'Modelos de Clasificación (U2)')
        pdf.parrafo("Predicción de Satisfacción Alta (4-5) vs Baja (1-3) con 4 algoritmos.")
        mejores = {}
        for modelo in ["Regresión Logística", "KNN", "SVM", "Árbol de Decisión"]:
            r = ms.entrenar_clasificacion(df, modelo)
            if not r.get("error"):
                mejores[modelo] = r
        if mejores:
            pdf.subtitulo('Comparación de Modelos')
            filas = []
            for m, r in mejores.items():
                filas.append((m, f"F1={r['metricas']['F1']:.3f}, Acc={r['metricas']['Accuracy']:.3f}"))
            pdf.tabla_metrica(filas)
            mejor_nombre = max(mejores, key=lambda k: mejores[k]['metricas']['F1'])
            mejor = mejores[mejor_nombre]
            pdf.parrafo(f"El mejor modelo fue {mejor_nombre} con F1-Score de {mejor['metricas']['F1']:.3f}.")
            pdf.agregar_figura(mejor["fig_cm"], f"Figura 2: Matriz de Confusión - {mejor_nombre}")
            if mejor.get("fig_roc"):
                pdf.agregar_figura(mejor["fig_roc"], f"Figura 3: Curva ROC - {mejor_nombre}")
            pdf.nota(
                f"El modelo {mejor_nombre} logró un F1 de {mejor['metricas']['F1']:.3f}, "
                f"lo que indica un buen equilibrio entre precisión y recall. "
                f"El AUC de {mejor['metricas'].get('AUC', 0):.3f} muestra la capacidad del modelo "
                f"para distinguir entre clientes satisfechos e insatisfechos."
            )

        # ============ 5. CLUSTERING ============
        pdf.titulo_capitulo('5', 'Aprendizaje No Supervisado (U3)')
        pdf.parrafo("Segmentación de clientes con K-Means, Clustering Jerárquico y DBSCAN.")
        res_km = mns.ejecutar_kmeans(df, n_clusters=3)
        if not res_km.get("error"):
            pdf.subtitulo('K-Means (k=3)')
            pdf.parrafo(f"Índice de Silhouette: {res_km['silhouette']:.4f}")
            pdf.agregar_figura(res_km["fig_codo"], "Figura 4: Método del Codo")
            pdf.agregar_figura(res_km["fig_pca"], "Figura 5: Clústeres K-Means (PCA 2D)")
            pdf.nota(
                f"Un Silhouette de {res_km['silhouette']:.3f} indica una separación "
                f"{'buena' if res_km['silhouette'] > 0.5 else 'moderada'} entre clústeres. "
                f"El método del codo sugiere que k=3 es un número razonable de segmentos."
            )
        res_jer = mns.ejecutar_jerarquico(df)
        if not res_jer.get("error"):
            pdf.agregar_figura(res_jer["fig_dendro"], "Figura 6: Dendrograma Jerárquico")

        # ============ 6. REFUERZO ============
        pdf.titulo_capitulo('6', 'Aprendizaje por Refuerzo (U4)')
        pdf.parrafo(
            "Se implementó Q-Learning en un entorno simulado donde un agente (barista) "
            "decide cuántos empleados asignar según la afluencia (Baja, Media, Alta)."
        )
        Q, historial = mr.ejecutar_q_learning(episodios=500, alpha=0.1, gamma=0.9)
        fig_ap, fig_q = mr.graficar_resultados(Q, historial)
        politica = mr.obtener_politica_optima(Q)
        pdf.agregar_figura(fig_ap, "Figura 7: Curva de Aprendizaje del Agente")
        pdf.agregar_figura(fig_q, "Figura 8: Matriz Q Final")
        pdf.subtitulo('Política Óptima Aprendida')
        for _, row in politica.iterrows():
            pdf.parrafo(f"- Estado {row['Estado (Afluencia)']}: {row['Mejor Acción']} (Q={row['Valor Q']})")
        pdf.nota(
            "El agente aprendió que en hora pico (afluencia Alta) conviene asignar "
            "3 baristas para maximizar la recompensa, mientras que en afluencia Baja "
            "basta con 1. Esto es consistente con la lógica del negocio."
        )

        # ============ 7. CONCLUSIONES ============
        pdf.titulo_capitulo('7', 'Conclusiones y Limitaciones')
        pdf.parrafo(
            "El proyecto demostró la aplicación exitosa de los 4 paradigmas de Aprendizaje "
            "Automático sobre un dataset tabular de 200+ registros. Se entrenaron modelos "
            "supervisados (Regresión y Clasificación), no supervisados (Clustering) y de "
            "Refuerzo (Q-Learning), todos con métricas de evaluación claras y reproducibles."
        )
        pdf.subtitulo('Limitaciones')
        pdf.parrafo(
            "- El dataset es pequeño (200 registros), lo que limita el uso de Deep Learning.\n"
            "- El Aprendizaje por Refuerzo se realizó en un entorno simulado, no en tiempo real.\n"
            "- No se aplicaron Transfer Learning ni GANs por ser datos tabulares, no imágenes/texto."
        )

        # ============ 8. REFERENCIAS ============
        pdf.titulo_capitulo('8', 'Referencias')
        pdf.parrafo(
            "1. Mitchell, T. (1997). Machine Learning. McGraw-Hill.\n"
            "2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.\n"
            "3. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT Press.\n"
            "4. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly."
        )

        # Guardar a bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            pdf.output(tmp.name)
            with open(tmp.name, 'rb') as f:
                pdf_bytes = f.read()
            os.unlink(tmp.name)

        return pdf_bytes, None

    except Exception as e:
        return None, f"Error al generar reporte: {e}"
```[cite: 2]

### Verificación
Reemplaza tu archivo `mod_reporte.py` con este código, reinicia Streamlit y vuelve a pulsar **Generar Reporte PDF**; el PDF debería compilarse sin el error del carácter `"•"`.