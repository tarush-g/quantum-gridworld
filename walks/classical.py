import numpy as np

def classical_distribution(P, T, start):
    """Cesaro time-average of the classical walk from `start` over T steps."""
 
    n = P.shape[0]
    p = np.zeros(n)
    p[start] = 1.0
 
    acc = np.zeros(n)
    for _ in range(T):
        acc += p
        p = p @ P
 
    pi = acc / T
    return pi
