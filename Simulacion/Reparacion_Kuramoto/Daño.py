import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numba import njit
from scipy.stats import gaussian_kde
from scipy.linalg import eigvals
from scipy.linalg import expm


# --- FUNCIONES DE DAÑO ---
def update_damage_matrix(h_matrix, adjacency_matrix):
    edges = np.argwhere(adjacency_matrix > 0)
    h_values = np.array([h_matrix[i, j] for i, j in edges])
    prob = h_values / np.sum(h_values)
    idx = np.random.choice(len(edges), p=prob)
    h_matrix[edges[idx][0], edges[idx][1]] += 1
    return h_matrix

def get_omega_matrix(h_matrix, adjacency_matrix, alpha):
    return (h_matrix.astype(float) ** (-alpha)) * adjacency_matrix

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

# --- FUNCIONES DE SINCRONIZACIÓN ---

def evolucion_fases(M, t, theta_0):
    """
    Calcula la evolución de las fases de cada nodo en el tiempo t.
    M: Matriz de acoplamiento (Omega)
    t: Escalar o array de tiempos
    theta_0: Vector de fases iniciales (condiciones iniciales)
    """
    L = Laplaciano(M)

    # Si t es un solo valor (escalar)
    if np.isscalar(t):
        matriz_evolucion = expm(-t * L)
        return matriz_evolucion @ theta_0

    # Si t es un array de tiempos, calculamos la fase para cada paso
    else:
        fases_historial = []
        for ti in t:
            matriz_evolucion = expm(-ti * L)
            fases_historial.append(matriz_evolucion @ theta_0)
        return np.array(fases_historial)

def order_parameter_evolucion(fases_t):
    return np.array([np.abs(np.mean(np.exp(1j * f))) for f in fases_t])

# --- FUNCIONES DE SIMULACION ---

def Simulacion_damage(A, alpha, T_repair, theta_0):
    """
    Solo corre la fase de daño y devuelve el estado final h, tau_0 y la funcionalidad.
    """
    N = A.shape[0]
    h = np.ones((N, N))
    tau_0 = None
    funcionalidad = np.zeros(T_repair + 1)  # <-- nuevo

    for T_step in range(T_repair + 1):
        Omega = get_omega_matrix(h, A, alpha)
        xi2 = get_second_eigenvalue_real(Omega)
        tau_T = 1.0 / xi2

        if T_step == 0:
            tau_0 = tau_T

        funcionalidad[T_step] = tau_0 / tau_T  # <-- nuevo

        if T_step < T_repair:
            h = update_damage_matrix(h, A)

    return h, tau_0, funcionalidad, Omega  # <-- nuevo

def Simulacion_damage_only(A, alpha, T_Max):
    """
    Solo acumula daño sobre h y devuelve Omega final.
    No calcula funcionalidad ni autovalores — mucho más rápido.
    """
    N = A.shape[0]
    h = np.ones((N, N))

    for T_step in range(T_Max):
        h = update_damage_matrix(h, A)

    Omega = get_omega_matrix(h, A, alpha)
    return Omega


#-------------------------------------------RED CON DOS COMUNIDADES-------------------------------------------

def Two_Communities_random(n, p):
    sizes = [n, n]  # dos comunidades
    # Probabilidades de conexión
    p_in = 1-p        # intra-comunidad
    p_out = p     # inter-comunidad
    # Matriz de probabilidades
    probs = [[p_in, p_out],[p_out, p_in]]
    # Generar red SBM
    G = nx.stochastic_block_model(sizes, probs, seed=42, directed=True) #Quitar seed si quieres redes aleatorias
    return G


# --- PARÁMETROS ---
num_simulations = 1000
alpha_damage = 1.0
T_Max = 100000
N = 10
G = Two_Communities_random(N, 0.1)
A = nx.to_numpy_array(G)

start_time_total = time.time()

# --- BUCLE DE SIMULACIONES ---
Omega_damage_list = []
for sim in range(num_simulations):
    Omega_damage = Simulacion_damage_only(A, alpha_damage, T_Max)
    Omega_damage_list.append(Omega_damage)

    if (sim + 1) % 10 == 0:
        print(f"  Simulación {sim+1}/{num_simulations} lista.")

print(f"\n--- Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos ---")

# --- GUARDAR ---
Omega_damage_array = np.array(Omega_damage_list)
A_origin = np.array(A)
np.save('Omega_damage_list.npy', Omega_damage_array)
np.save('A_origin.npy',A_origin)
