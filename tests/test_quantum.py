import numpy as np
import pytest 
from quantum import landau_levels


#Loop over n = 0..5, assert each eigenvalue is within 1e-3 of n + 0.5
def test_landau_ladder():
    _, evals, _ = landau_levels()
    for n, E in enumerate(evals):
        assert abs(E - (n+0.5)) < 1e-3


#eigenvalues at x0=0 and x0=2 agree to 1e-8
def test_degeneracy():
    _, e0, _ = landau_levels(0.0)
    _, e2, _ = landau_levels(2.0)
    np.testing.assert_allclose(e2, e0, atol=1e-8)


#Two grid sizes, N and 2N, over the same box, the error in E₅ shrinks by roughly 4×
#E₅ is the 6th eigenvalue (n=5), which should be exactly 5.5
#error is abs(evals[5] - 5.5)
#use n=5 rather than n=0 because higher states have larger truncation error, so the convergence signal is clearer
def test_convergence():
    _, evals1, _ = landau_levels(N=500)
    _, evals2, _ = landau_levels(N=1000)
    err1 = abs(evals1[5] - 5.5)
    err2 = abs(evals2[5] - 5.5)
    ratio = err1 / err2
    assert 3.5 < ratio < 4.5


#crossing count will pick up noise in the exponential tails where ψ is essentially zero but flickers sign
#test in the meaningful region
@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_nodes(n):
    _, _, evecs = landau_levels()
    psi = evecs[:, n]
    mask = np.abs(psi) > 0.01 * np.abs(psi).max()
    signs = np.sign(psi[mask])
    crossings = np.sum(np.abs(np.diff(signs)) != 0) 
    assert crossings == n


#bridge consistency --> assert sqrt(2n+1) squared over 2 equals n + 0.5
@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_bridge(n):
    r_n = np.sqrt(2*n+1)
    assert abs(0.5 * r_n**2 - (n+0.5)) < 1e-12 #classical and quantum bridge check 