"""Gráficas obligatorias del temario: histogramas, boxplots, barras y ajustes."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


def _fig():
    fig, ax = plt.subplots(figsize=(8, 4.5))
    return fig, ax


def plot_hist(series, title, xlabel, bins=20):
    fig, ax = _fig()
    s = pd.to_numeric(series, errors="coerce").dropna()
    ax.hist(s, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    ax.grid(alpha=0.3)
    return fig


def plot_box(series, title, ylabel):
    fig, ax = _fig()
    s = pd.to_numeric(series, errors="coerce").dropna()
    ax.boxplot(s, vert=True, patch_artist=True,
               boxprops=dict(facecolor="#DD8452"))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([1])
    ax.set_xticklabels([ylabel])
    ax.grid(alpha=0.3)
    return fig


def plot_box_by_group(df, value_col, group_col, title, ylabel):
    """Boxplot comparativo por grupo (compatible con Matplotlib 3.7–3.9+)."""
    fig, ax = _fig()
    grouped = df.groupby(group_col)
    groups = [g[value_col].dropna().values for _, g in grouped]
    labels = [str(k) for k, _ in grouped]

    ax.boxplot(groups, patch_artist=True,
               boxprops=dict(facecolor="#55A868"))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return fig


def plot_bars(df, col, title):
    fig, ax = _fig()
    vc = df[col].fillna("(Nulo)").value_counts()
    ax.bar(vc.index.astype(str), vc.values, color="#C44E52", edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", alpha=0.3)
    return fig


def plot_exponential_fit(series, title="Ajuste Exponencial"):
    fig, ax = _fig()
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    ax.hist(s, bins=25, density=True, alpha=0.6,
            color="#4C72B0", edgecolor="white", label="Datos")
    lam = 1.0 / s.mean()
    x = np.linspace(0, s.max(), 300)
    ax.plot(x, stats.expon.pdf(x, scale=1 / lam),
            "r-", lw=2, label=f"Exp(λ={lam:.3f})")

    ks_stat, p = stats.kstest(s, stats.expon(scale=1 / lam).cdf)

    ax.set_title(f"{title}\nKS={ks_stat:.3f}, p={p:.3f}")
    ax.set_xlabel("Tiempo_Espera_Min")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_normal_fit(series, title="Ajuste Normal"):
    fig, ax = _fig()
    s = pd.to_numeric(series, errors="coerce").dropna()
    s = s[s > 0]
    ax.hist(s, bins=25, density=True, alpha=0.6,
            color="#55A868", edgecolor="white", label="Datos")
    mu, sigma = s.mean(), s.std(ddof=1)
    x = np.linspace(s.min(), s.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma),
            "r-", lw=2, label=f"N(μ={mu:.2f}, σ={sigma:.2f})")

    ks_stat, p = stats.kstest(s, stats.norm(loc=mu, scale=sigma).cdf)

    ax.set_title(f"{title}\nKS={ks_stat:.3f}, p={p:.3f}")
    ax.set_xlabel("Tiempo_Servicio_Min")
    ax.set_ylabel("Densidad")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_box_by_binary(df, value_col, binary_col, title, ylabel):
    """Boxplot comparativo para una variable binaria (Sí/No)."""
    fig, ax = _fig()

    order = [v for v in ["No", "Sí"] if v in df[binary_col].unique()]
    grouped = df.groupby(binary_col)
    groups = [grouped.get_group(k)[value_col].dropna().values for k in order]
    labels = order

    bp = ax.boxplot(groups, patch_artist=True, widths=0.5)
    colors = ["#4C72B0", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors[:len(groups)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(binary_col)
    ax.grid(alpha=0.3, axis="y")

    for i, g in enumerate(groups):
        if len(g) > 0:
            ax.text(i + 1, ax.get_ylim()[1] * 0.95,
                    f"μ={g.mean():.1f}", ha="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="gray", alpha=0.8))

    return fig


def plot_grouped_bars(df, value_col, group_col, title, ylabel="Conteo"):
    """
    Barras agrupadas: distribución de value_col (categórica) por group_col.
    Compatible con columnas Int64 que contienen pd.NA.
    """
    fig, ax = _fig()

    # Convertir a object primero, luego reemplazar nulos, luego a str
    valores = df[value_col].astype(object)
    valores = valores.where(valores.notna(), "(Nulo)")
    valores = valores.astype(str)

    ct = pd.crosstab(df[group_col], valores).sort_index()

    cmap = plt.get_cmap("viridis")
    n_vals = ct.shape[1]
    colors = [cmap(i / max(n_vals - 1, 1)) for i in range(n_vals)]

    ct.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.8)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(group_col)
    ax.legend(title=value_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=0)

    return fig