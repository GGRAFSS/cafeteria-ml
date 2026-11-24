"""
Módulo Unidad 4: Aprendizaje por Refuerzo
- Implementación de Q-Learning (Ecuación de Bellman)
- Entorno simulado: Agente Barista que decide cuántos empleados asignar
- Recompensa: Maximizar ganancias, minimizar tiempo de espera
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EntornoCafeteria:
    """
    Simulador simple de la cafetería.
    Estados: Nivel de ocupación (0 = Baja, 1 = Media, 2 = Alta)
    Acciones: Número de baristas a asignar (1, 2, o 3)
    Recompensa: Ganancia estimada (ingresos - costo de empleados)
    """
    def __init__(self):
        self.num_estados = 3
        self.num_acciones = 3
        self.estado = 0
        self.paso = 0
        self.max_pasos = 50
        
        # Matriz de recompensas (Estado x Acción)
        # Estado 0 (Baja): 1 barista da +10, 2 dan +5, 3 dan -5
        # Estado 1 (Media): 1 barista da +5, 2 dan +15, 3 dan +10
        # Estado 2 (Alta): 1 barista da -10, 2 dan +10, 3 dan +20
        self.R = np.array([
            [10, 5, -5],
            [5, 15, 10],
            [-10, 10, 20]
        ])
        
    def reset(self):
        self.estado = np.random.choice([0, 1, 2])
        self.paso = 0
        return self.estado
    
    def step(self, accion):
        # La recompensa se basa en la matriz
        recompensa = self.R[self.estado, accion]
        
        # El siguiente estado depende de la afluencia (simulado)
        # Si el estado era alto, probablemente siga alto o baje a medio
        if self.estado == 0:
            siguiente_estado = np.random.choice([0, 1], p=[0.7, 0.3])
        elif self.estado == 1:
            siguiente_estado = np.random.choice([0, 1, 2], p=[0.2, 0.6, 0.2])
        else:
            siguiente_estado = np.random.choice([1, 2], p=[0.4, 0.6])
            
        self.estado = siguiente_estado
        self.paso += 1
        done = self.paso >= self.max_pasos
        return siguiente_estado, recompensa, done

def ejecutar_q_learning(episodios=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    env = EntornoCafeteria()
    # Matriz Q inicializada en ceros
    Q = np.zeros((env.num_estados, env.num_acciones))
    
    historial_recompensas = []
    
    for ep in range(episodios):
        estado = env.reset()
        recompensa_total = 0
        done = False
        
        while not done:
            # Estrategia Epsilon-Greedy (Exploración vs Explotación)
            if np.random.uniform(0, 1) < epsilon:
                accion = np.random.choice(env.num_acciones) # Explorar
            else:
                accion = np.argmax(Q[estado, :]) # Explotar (mejor acción conocida)
            
            siguiente_estado, recompensa, done = env.step(accion)
            
            # --- ECUACIÓN DE BELLMAN ---
            # Q(s,a) = Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
            max_q_siguiente = np.max(Q[siguiente_estado, :])
            Q[estado, accion] = Q[estado, accion] + alpha * (
                recompensa + gamma * max_q_siguiente - Q[estado, accion]
            )
            
            estado = siguiente_estado
            recompensa_total += recompensa
            
        historial_recompensas.append(recompensa_total)
        
    return Q, historial_recompensas

def graficar_resultados(Q, historial):
    # --- Gráfica 1: Evolución del Aprendizaje ---
    fig_aprendizaje, ax1 = plt.subplots(figsize=(8, 4))
    # Suavizar la curva con media móvil
    ventana = 20
    suavizado = pd.Series(historial).rolling(window=ventana).mean()
    ax1.plot(historial, alpha=0.3, color="gray", label="Recompensa por Episodio")
    ax1.plot(suavizado, color="blue", lw=2, label=f"Media Móvil ({ventana} ep.)")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Recompensa Acumulada")
    ax1.set_title("Curva de Aprendizaje del Agente (Q-Learning)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Gráfica 2: Matriz Q Final ---
    fig_q, ax2 = plt.subplots(figsize=(6, 4))
    im = ax2.imshow(Q, cmap="Blues")
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(["1 Barista", "2 Baristas", "3 Baristas"])
    ax2.set_yticklabels(["Baja", "Media", "Alta"])
    ax2.set_xlabel("Acción (Empleados)")
    ax2.set_ylabel("Estado (Afluencia)")
    ax2.set_title("Matriz Q Final (Valor de cada Acción)")
    
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{Q[i, j]:.1f}", ha="center", va="center",
                     color="white" if Q[i, j] > Q.max()/2 else "black", fontsize=12)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    return fig_aprendizaje, fig_q

def obtener_politica_optima(Q):
    """Extrae la mejor acción para cada estado."""
    politica = []
    estados_nombres = ["Baja", "Media", "Alta"]
    acciones_nombres = ["1 Barista", "2 Baristas", "3 Baristas"]
    
    for s in range(Q.shape[0]):
        mejor_accion = np.argmax(Q[s, :])
        politica.append({
            "Estado (Afluencia)": estados_nombres[s],
            "Mejor Acción": acciones_nombres[mejor_accion],
            "Valor Q": round(Q[s, mejor_accion], 2)
        })
    return pd.DataFrame(politica)