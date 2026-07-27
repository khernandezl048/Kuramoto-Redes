import numpy as np
import networkx as nx
import time
from scipy.linalg import expm

# ------------------------------- FUNCIONES DE CALCULO DE AUTOVALORES -------------------------------

def Laplaciano(M):
    S = M.sum(axis=1)
    L = np.diag(S) - M
    return L

def get_second_eigenvalue_real(M):
    L = Laplaciano(M)
    evals = np.linalg.eigvals(L)
    real_parts = np.sort(np.real(evals))
    return real_parts[1]

# ------------------------------- FUNCIONES DE REPARACIÓN -------------------------------

def _seleccionar_enlace2(Omega_matrix, strategy):
    """
    Devuelve las coordenadas (i, j) del enlace a reparar según strategy.
    Retorna None si no hay enlaces dañados.

    strategy = 'high_betweenness' : mayor edge betweenness
    strategy = 'low_betweenness'  : menor edge betweenness
    """

    edges = np.argwhere(Omega_matrix > 0)
    Omega_values = np.array([Omega_matrix[i, j] for i, j in edges])

    mask = Omega_values < 1.0
    if not np.any(mask):
        return None

    edges_reparables = edges[mask]

    # ---------------------------------------------------------
    # Construir red ponderada usando distancia = 1/Omega
    # ---------------------------------------------------------

    G = nx.Graph()

    N = Omega_matrix.shape[0]

    for i in range(N):
        for j in range(i + 1, N):

            omega = Omega_matrix[i, j]

            if omega > 0:

                G.add_edge(
                    i,
                    j,
                    distance=1.0 / omega
                )

    # Edge betweenness ponderada
    betweenness = nx.edge_betweenness_centrality(
        G,
        weight='distance'
    )

    # ---------------------------------------------------------
    # Obtener betweenness de los enlaces reparables
    # ---------------------------------------------------------

    bw_values = np.array([
        betweenness.get((i, j), betweenness.get((j, i), 0.0))
        for i, j in edges_reparables
    ])

    if strategy == 'high_betweenness':
        idx = np.argmax(bw_values)

    elif strategy == 'low_betweenness':
        idx = np.argmin(bw_values)

    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")

    return edges_reparables[idx]

def Simulacion_repair_deterministic2(A, Omega_damage, T_repair, delta_r, strategy='best'):
    tau_0 = 1.0 / get_second_eigenvalue_real(A)
    Omega = Omega_damage.copy()
    funcionalidad = np.zeros(T_repair + 1)

    F_max = -np.inf
    Omega_at_Fmax = Omega.copy()

    for T_step in range(T_repair + 1):
        xi2 = get_second_eigenvalue_real(Omega)
        tau_T = 1.0 / xi2
        F_s = tau_0 / tau_T
        funcionalidad[T_step] = F_s

        if F_s > F_max:
            F_max = F_s
            Omega_at_Fmax = Omega.copy()

        if T_step < T_repair:
            # Recalcular betweenness en cada paso
            enlace_actual = _seleccionar_enlace2(Omega, strategy)

            if enlace_actual is None:
                funcionalidad[T_step + 1:] = funcionalidad[T_step]
                break

            i, j = enlace_actual
            Omega[i, j] = min(Omega[i, j] + delta_r, 1.0)

    return funcionalidad, Omega_at_Fmax, F_max

# ---------------------------------------- CARGAR DATOS ----------------------------------------
Omega_damage_array = np.load('../Omega_damage_list.npy')
Omega_damage_list = [Omega_damage_array[i] for i in range(Omega_damage_array.shape[0])]
A = np.load('../A_origin.npy')

# ---------------------------------------- SIMULACION ----------------------------------------

#Tipo_Simulacion = "strategy"
Tipo_Simulacion = "delta_r"

# ----------------------------------------------------
# COMPARAR ESTRATEGIAS
# ----------------------------------------------------

if Tipo_Simulacion == "strategy":

    T_repair = 25000
    delta_r = 0.01

    strategies = [
        'high_betweenness',
        'low_betweenness'
    ]

    avg_repair = {s: np.zeros(T_repair + 1) for s in strategies}

    start_time_total = time.time()

    for sim, Omega_damage in enumerate(Omega_damage_list):

        for s in strategies:

            f_repair, _, _ = Simulacion_repair_deterministic2(
                A,
                Omega_damage,
                T_repair,
                delta_r,
                strategy=s
            )

            avg_repair[s] += f_repair

        if (sim + 1) % 10 == 0:
            print(f"\n --- Simulación {sim+1}/{len(Omega_damage_list)} lista ---")
            print(f"--- Tiempo parcial: {(time.time() - start_time_total)/60:.2f} minutos ---")

    for s in strategies:
        avg_repair[s] /= len(Omega_damage_list)

    print(f"\n--- Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos ---")

    np.save(
        'F_edge_strategy.npy',
        avg_repair
    )

elif Tipo_Simulacion == "delta_r":

    T_repair = 25000

    deltas_r = np.linspace(0.01, 0.10, 10)

    avg_repair = {
        d: np.zeros(T_repair + 1)
        for d in deltas_r
    }

    start_time_total = time.time()

    for sim, Omega_damage in enumerate(Omega_damage_list):

        for d in deltas_r:

            f_repair, _, _ = Simulacion_repair_deterministic2(
                A,
                Omega_damage,
                T_repair,
                d,
                strategy='low_betweenness'   # o high_betweenness---low_betweenness
            )

            avg_repair[d] += f_repair

        if (sim + 1) % 10 == 0:
            print(f"\n --- Simulación {sim+1}/{len(Omega_damage_list)} lista ---")
            print(f"--- Tiempo parcial: {(time.time() - start_time_total)/60:.2f} minutos ---")

    for d in deltas_r:
        avg_repair[d] /= len(Omega_damage_list)

    print(f"\n--- Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos ---")

    np.save('F_edge_deltar_low_betweenness.npy',avg_repair)