import numpy as np 

def divergence(field, point, which=1, h=1e-5): #which=1 is default and B, which=0 is E
    total = 0.0
    for i in range(3):
        p = np.array(point, dtype=float)
        p[i] += h
        v_plus = field(p, 0.0)[which]
        p = np.array(point, dtype=float)
        p[i] -= h
        v_minus = field(p, 0.0)[which]
        total += (v_plus[i] - v_minus[i]) / (2 * h)
    return total 