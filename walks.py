import numpy as np

def classical(P, steps, start):
    """Time averaged classical walk distribution."""
 
    n = P.shape[0]
    p = np.zeros(n)
    p[start] = 1.0
 
    acc = np.zeros(n)
    for _ in range(steps):
        acc += p
        p = p @ P
 
    pi = acc / steps
    return pi


def szegedy(P, steps, start):
    """Time averaged quantum walk distribution"""

    n = P.shape[0]
    P = np.sqrt(P)

    psi = np.zeros((n, n), dtype=complex)
    psi[start] = P[start]

    probs = np.zeros((n, n))
    for _ in range(steps):
        probs += np.abs(psi) ** 2
        coeff = (P * psi).sum(axis=1)
        psi = (2 * coeff[:, None] * P - psi).T # reflection and swap

    return probs / steps