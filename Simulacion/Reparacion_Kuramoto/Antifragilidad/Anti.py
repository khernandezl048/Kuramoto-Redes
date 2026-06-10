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
def Damage(beta, A, i):
    """
    Daña el enlace a->b con factor (1-beta) y recalcula W(Omega*)
    con normalización por fila.
    a: nodo origen del enlace dañado
    b: nodo destino del enlace dañado
    """
    Omega_star = A.copy()

    edges = np.argwhere(A > 0)
    
    idx = edges[i]
    a = idx[0]
    b = idx[1]

    # Solo modificar el enlace a->b
    Omega_star[a, b] = (1 - beta) * A[a, b]

    # Normalización por fila: W_ij = Omega*_ij / sum_l(Omega*_il)
    row_sums = Omega_star.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # evitar división por cero
    W = Omega_star / row_sums

    return W, idx

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

# --- FUNCIONES DE ANTIFRAGILIDAD ---

def Antifragilidad(Funcionalidad, beta, n_points=4):
    
    coef = np.polyfit(beta[:n_points], Funcionalidad[:n_points], 1)
    return coef[0]
    


# --- FUNCIONES DE SIMULACION ---

def Simulacion(beta, A, enlace):
    W_0, idx = Damage(0, A, enlace)
    tau_0 = 1.0 / get_second_eigenvalue_real(W_0)
    funcionalidad = np.zeros(len(beta))
    for i in range(len(beta)):
        W , _ = Damage(beta[i], A, enlace)
        xi2 = get_second_eigenvalue_real(W)
        tau_B = 1.0 / xi2
        funcionalidad[i] = tau_0 / tau_B

    return funcionalidad, idx


# ---------------------------------------- CARGAR DATOS ----------------------------------------
A = np.load('../A_origin.npy')

# A = (A > 0).astype(float)#opcional--Si la red esta dañada arregla todos los enlaces para el analisis

# ---------------------------------------- PARÁMETROS ----------------------------------------

delta_B = 0.0001
Betas = np.arange(0, 0.01 + delta_B, delta_B)

# Lista de enlaces existentes
edges = np.argwhere(A > 0)

Resultados = []

start_time_total = time.time()

# ---------------------------------------- BUCLE PRINCIPAL ----------------------------------------

for enlace in range(len(edges)):

    Funcionalidad, idx = Simulacion(Betas, A, enlace)

    Lambda = Antifragilidad(Funcionalidad, Betas, n_points=4)

    Resultados.append([
        idx[0],   # nodo origen
        idx[1],   # nodo destino
        Lambda
    ])

    if enlace % 100 == 0:
        print(f"Procesados {enlace}/{len(edges)} enlaces")

# Convertir a numpy
Resultados = np.array(Resultados)

# Guardar
np.save("Datos_Antifragilidad.npy", Resultados)

print(Resultados[:10])

print(f"\n--- Tiempo total: {(time.time() - start_time_total)/60:.2f} minutos ---")

