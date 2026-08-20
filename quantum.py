import numpy as np 
from scipy.linalg import eigh_tridiagonal


def landau_levels(x0=0.0, L=20.0, N=2000, n_levels=6):
    x = np.linspace(-L/2, L/2, N) #grid from -L/2 to L/2
    dx = x[1] - x[0] #L/ N 
    #the diagonal, array of length N
    d = 1/dx**2 + 0.5*(x-x0)**2 #vectorized, d is an array 
    #the off diagonal of length N-1, every entry is the same value 
    e = np.full(N-1, -0.5/dx**2)
    #return first n_levels eigenvalues
    evals, evecs = eigh_tridiagonal(d, e, select='i', select_range=(0, n_levels-1)) 
    return x, evals, evecs
    #x is the spatial grid, 2000 points from −10 to +10, the positions where the wavefunction is evaluated


if __name__ == "__main__":
    _, evals, evecs = landau_levels() 
    for n, E in enumerate(evals):
        print(f"n={n}  E={E:.6f}  (ecpect {n + 0.5})")
        #with finite-difference truncation error