import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numba import njit
from scipy.stats import gaussian_kde
from scipy.linalg import eigvals
from scipy.linalg import expm


##--------------------------FUNCIONES------------------------------
def deterministic_repair_Omega(Omega_matrix, delta_r, strategy='best'):
    """
    Repara determinísticamente un enlace dañado.
    strategy = 'best'  : repara el enlace en mejor estado (mayor Omega_ij)
    strategy = 'worst' : repara el enlace en peor estado  (menor Omega_ij)
    """
    edges = np.argwhere(Omega_matrix > 0)
    Omega_values = np.array([Omega_matrix[i, j] for i, j in edges])

    mask = Omega_values < 1.0
    if not np.any(mask):
        return Omega_matrix

    edges_reparables = edges[mask]
    Omega_reparables = Omega_values[mask]

    if strategy == 'best':
        idx = np.argmax(Omega_reparables)
    elif strategy == 'worst':
        idx = np.argmin(Omega_reparables)

    i, j = edges_reparables[idx]
    Omega_matrix[i, j] = min(Omega_matrix[i, j] + delta_r, 1.0)

    return Omega_matrix


def Simulacion_repair_deterministic(A, Omega_damage, T_repair, delta_r, strategy='best'):
    tau_0 = 1.0 / get_second_eigenvalue_real(A)
    Omega = Omega_damage.copy()
    funcionalidad = np.zeros(T_repair + 1)

    for T_step in range(T_repair + 1):
        xi2 = get_second_eigenvalue_real(Omega)
        tau_T = 1.0 / xi2
        funcionalidad[T_step] = tau_0 / tau_T

        if T_step < T_repair:
            Omega = deterministic_repair_Omega(Omega, delta_r, strategy)

    return funcionalidad, Omega

# --- FUNCIONES DE CALCULO DE AUTOVALORES ---

def Laplaciano(M):
    S = M.sum(axis=1)

    # Construir el Laplaciano: L = diag(S) - M
    # np.diag(S) crea la matriz con S_i en la diagonal principal (S_i * delta_ij)
    L = np.diag(S) - M

    return L

def get_second_eigenvalue_real(M):

    L = Laplaciano(M)

    #Calcular autovalores
    evals = np.linalg.eigvals(L)

    # Extraer partes reales y ordenar
    real_parts = np.sort(np.real(evals))

    # Retornamos el segundo autovalor
    return real_parts[1]


#--------------------DATOS DE LAS REDES DAÑADAS------------------

# --- CARGAR datos ---
Omega_damage_array = np.load('Omega_damage_list.npy')
Omega_damage_list = [Omega_damage_array[i] for i in range(Omega_damage_array.shape[0])]
A = np.load('A_origin.npy')
G = nx.from_numpy_array(A)

#-------------------------SIMULACION-------------------

# --- PARÁMETROS ---
T_repair = 20000
deltas_r = np.linspace(0.01, 0.1, 10)

# --- ACUMULADORES ---
avg_repair = {d: np.zeros(T_repair + 1) for d in deltas_r}

start_time_total = time.time()

# --- BUCLE DE SIMULACIONES ---
for sim, Omega_damage in enumerate(Omega_damage_list):
    for d in deltas_r:
        f_repair, _ = Simulacion_repair_deterministic(A, Omega_damage, T_repair, d, strategy='best')
        avg_repair[d] += f_repair

    if (sim + 1) % 10 == 0:
        print(f"  Simulación {sim+1}/{len(Omega_damage_list)} lista.")

# --- PROMEDIOS ---
for d in deltas_r:
    avg_repair[d] /= len(Omega_damage_list)

print(f"\n--- Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos ---")

np.save('F_Deterministic1.npy',avg_repair) ## GUARDAR DATOS