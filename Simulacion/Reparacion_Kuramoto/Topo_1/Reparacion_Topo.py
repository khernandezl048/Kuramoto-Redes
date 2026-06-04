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

def es_inter_comunidad(i, j, comunidad_0, comunidad_1):
    return (i in comunidad_0 and j in comunidad_1) or \
           (i in comunidad_1 and j in comunidad_0)


def _seleccionar_enlace_inter(Omega_matrix, N_com=10):
    edges = np.argwhere(Omega_matrix > 0)
    Omega_values = np.array([Omega_matrix[i, j] for i, j in edges])

    mask = Omega_values < 1.0
    if not np.any(mask):
        return None

    edges_reparables = edges[mask]
    Omega_reparables = Omega_values[mask]

    comunidad_0 = set(range(N_com))
    comunidad_1 = set(range(N_com, 2 * N_com))

    mask_inter = np.array([
        es_inter_comunidad(i, j, comunidad_0, comunidad_1)
        for i, j in edges_reparables
    ])

    if np.any(mask_inter):
        edges_inter = edges_reparables[mask_inter]
        Omega_inter = Omega_reparables[mask_inter]
        idx = np.argmax(Omega_inter)
        return edges_inter[idx]
    else:
        idx = np.argmax(Omega_reparables)
        return edges_reparables[idx]


def Simulacion_repair_inter_first(A, Omega_damage, T_repair, delta_r, N_com=10):
    tau_0 = 1.0 / get_second_eigenvalue_real(A)
    Omega = Omega_damage.copy()
    funcionalidad = np.zeros(T_repair + 1)

    F_max = -np.inf
    Omega_at_Fmax = Omega.copy()

    enlace_actual = _seleccionar_enlace_inter(Omega, N_com)

    for T_step in range(T_repair + 1):
        xi2 = get_second_eigenvalue_real(Omega)
        tau_T = 1.0 / xi2
        F_s = tau_0 / tau_T
        funcionalidad[T_step] = F_s

        if F_s > F_max:
            F_max = F_s
            Omega_at_Fmax = Omega.copy()

        if T_step < T_repair:
            if enlace_actual is None:
                funcionalidad[T_step + 1:] = funcionalidad[T_step]
                break

            i, j = enlace_actual
            Omega[i, j] = min(Omega[i, j] + delta_r, 1.0)

            if Omega[i, j] >= 1.0:
                enlace_actual = _seleccionar_enlace_inter(Omega, N_com)

    return funcionalidad, Omega_at_Fmax, F_max


# ---------------------------------------- CARGAR DATOS ----------------------------------------
Omega_damage_array = np.load('../Omega_damage_list.npy')
Omega_damage_list = [Omega_damage_array[i] for i in range(Omega_damage_array.shape[0])]
A = np.load('../A_origin.npy')

# ---------------------------------------- SIMULACION ----------------------------------------

T_repair = 25000
N_com = 10

deltas_r = np.linspace(0.01, 0.1, 10)

avg_repair = {d: np.zeros(T_repair + 1) for d in deltas_r}

start_time_total = time.time()

for sim, Omega_damage in enumerate(Omega_damage_list[0:1]):

    for d in deltas_r:

        f_repair, Omega_Ver, _ = Simulacion_repair_inter_first(
            A,
            Omega_damage,
            T_repair,
            d,
            N_com=N_com
        )

        avg_repair[d] += f_repair

    if (sim + 1) % 10 == 0:
        print(f"\n --- Simulación {sim+1}/{len(Omega_damage_list[0:1])} lista. ---")
        print(f"--- Tiempo parcial: {(time.time() - start_time_total)/60:.2f} minutos ---")

# Promediar
for d in deltas_r:
    avg_repair[d] /= len(Omega_damage_list[0:1])

print(f"\n--- Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos ---")

np.save('F_inter_first_best_1.npy', avg_repair)
