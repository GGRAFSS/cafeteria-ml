"""
Módulo Unidad 3: Aprendizaje No Supervisado
- Distancias (Euclidiana y Jaccard)
- Clustering: K-Means, Jerárquico y DBSCAN
- Reducción de Dimensionalidad: PCA para visualización
- Validación: Índice de Silhouette
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

def _preparar_datos(df):
    """Limpia, codifica variables categóricas y escala para clustering."""
    df_clean = df.dropna(subset=["Consumo_Total_MXN", "Tiempo_Espera_Min", "Tiempo_Servicio_Min"]).copy()
    
    # Codificar categóricas
    cat_cols = ["Tipo_Cliente", "Metodo_Pago", "Es_Hora_Pico"]
    for col in cat_cols:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str).fillna("Nulo"))
            
    # Seleccionar features numéricas para clustering (evitamos ID y Satisfacción)
    features = ["Hora_Llegada", "Tiempo_Espera_Min", "Tiempo_Servicio_Min", 
                "Consumo_Total_MXN", "Tipo_Cliente", "Metodo_Pago", "Es_Hora_Pico"]
    
    X = df_clean[[f for f in features if f in df_clean.columns]].values
    
    # Escalar (crucial para distancias)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df_clean

# ============================================================
# 1. CÁLCULO DE DISTANCIAS (Euclidiana y Jaccard)
# ============================================================
def calcular_distancias(df, id1, id2):
    if df.empty:
        return {"error": "No hay datos."}
    
    cliente1 = df[df["ID_Cliente"] == id1]
    cliente2 = df[df["ID_Cliente"] == id2]
    
    if cliente1.empty or cliente2.empty:
        return {"error": "Uno de los IDs no existe."}

    # --- Distancia Euclidiana (Variables numéricas) ---
    cols_num = ["Hora_Llegada", "Tiempo_Espera_Min", "Tiempo_Servicio_Min", "Consumo_Total_MXN"]
    p = cliente1[cols_num].values[0]
    q = cliente2[cols_num].values[0]
    dist_euclidiana = np.sqrt(np.sum((p - q)**2))
    
    # --- Distancia de Jaccard (Similitud categórica) ---
    # Convertimos las categóricas en sets de palabras
    cat_cols = ["Tipo_Cliente", "Metodo_Pago", "Es_Hora_Pico"]
    set1 = set([f"{col}_{cliente1[col].values[0]}" for col in cat_cols])
    set2 = set([f"{col}_{cliente2[col].values[0]}" for col in cat_cols])
    
    interseccion = len(set1.intersection(set2))
    union = len(set1.union(set2))
    jaccard_similitud = interseccion / union if union > 0 else 0
    
    return {
        "dist_euclidiana": dist_euclidiana,
        "jaccard_similitud": jaccard_similitud,
        "jaccard_distancia": 1 - jaccard_similitud,
        "error": None
    }

# ============================================================
# 2. K-MEANS (Partición)
# ============================================================
def ejecutar_kmeans(df, n_clusters=3):
    X_scaled, df_clean = _preparar_datos(df)
    if len(X_scaled) < n_clusters:
        return {"error": "Datos insuficientes para el número de clústeres seleccionado."}

    # Método del Codo
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        
    fig_codo, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(K_range, inertias, 'bx-')
    ax1.set_xlabel('Número de Clústeres (k)')
    ax1.set_ylabel('Inercia')
    ax1.set_title('Método del Codo')
    ax1.grid(alpha=0.3)

    # Entrenamiento final
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0

    # PCA para visualización en 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig_pca, ax2 = plt.subplots(figsize=(6, 4))
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolor='k')
    ax2.set_title(f'Clústeres K-Means (k={n_clusters})')
    ax2.set_xlabel('Componente Principal 1')
    ax2.set_ylabel('Componente Principal 2')
    plt.colorbar(scatter, ax=ax2, label='Clúster')
    ax2.grid(alpha=0.3)

    return {
        "fig_codo": fig_codo,
        "fig_pca": fig_pca,
        "silhouette": sil_score,
        "error": None
    }

# ============================================================
# 3. JERÁRQUICO (Aglomerativo)
# ============================================================
def ejecutar_jerarquico(df):
    X_scaled, df_clean = _preparar_datos(df)
    if len(X_scaled) < 3:
        return {"error": "Datos insuficientes."}

    # Dendrograma
    linked = linkage(X_scaled, method='ward')
    fig_dendro, ax1 = plt.subplots(figsize=(10, 5))
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True, ax=ax1)
    ax1.set_title('Dendrograma — Clustering Jerárquico')
    ax1.set_xlabel('Índice del Cliente')
    ax1.set_ylabel('Distancia')

    return {
        "fig_dendro": fig_dendro,
        "error": None
    }

# ============================================================
# 4. DBSCAN (Densidad)
# ============================================================
def ejecutar_dbscan(df, eps=0.5, min_samples=5):
    X_scaled, df_clean = _preparar_datos(df)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # PCA para visualización
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig_pca, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='plasma', alpha=0.6, edgecolor='k')
    ax.set_title(f'DBSCAN (Clústeres: {n_clusters}, Ruido: {n_noise})')
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    plt.colorbar(scatter, ax=ax, label='Clúster')
    ax.grid(alpha=0.3)

    return {
        "fig_pca": fig_pca,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "error": None
    }