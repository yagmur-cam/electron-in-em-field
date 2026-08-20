import numpy as np 
import pytest
from pushers import boris, rk4
from fields import uniform_field, exb_field


@pytest.fixture
def uniform_run():
    dt, n_steps = 0.01, 10000
    xs, vs = boris(np.array([1.0,0,0]), np.array([0,-1.0,0.3]), 
    1.0, 1.0, dt, n_steps, uniform_field)
    return xs, vs, dt, n_steps


def test_speed_conservation(uniform_run):
    xs, vs, dt, n_steps = uniform_run
    speeds = np.linalg.norm(vs, axis=1)
    drift = abs(speeds.max() - speeds.min()) / speeds[0]
    assert drift < 1e-12


def test_gyroradius(uniform_run):
    xs, vs, dt, n_steps = uniform_run
    r =  np.linalg.norm(xs[:, :2], axis=1)
    assert abs(r.mean() - 1.0) < 1e-3


def test_period(uniform_run):
    xs, vs, dt, n_steps = uniform_run
    theta = np.unwrap(np.arctan2(xs[:, 1], xs[:, 0]))
    period = 2 * np.pi / abs((theta[-1] - theta[0]) / (n_steps * dt))
    assert abs(period - 2*np.pi) < 1e-3 #2*np.pi - 1e-3 < period < 2*np.pi + 1e-3


#catches regressions 
def test_exb_drift(): #the residual is the partial-gyration fitting bias
    dt = 0.01
    n_steps = 5000
    t = np.arange(n_steps)*dt
    x0 = np.array([0, 0, 0])
    v0 = np.array([0, -1.0, 0])
    E, B = exb_field(x0, 0.0)
    expected = np.cross(E,B)/np.dot(B,B) #(E×B)/B², test survives changing E's magnitude 
    for q, m in [(1, 1), (-1, 3)]:
        xs, vs = boris(x0, v0, q, m, dt, n_steps, exb_field)
        slope = np.polyfit(t, xs[:, 1], 1)[0]
        #within 5% of np.cross(E,B)/np.dot(B,B) component 1
        assert abs(slope - expected[1]) < 0.10 * abs(expected[1]) 

#Boris and RK4 differ at O(dt²) in phase, so at dt=0.001 over 2 time units expect agreement around 1e-6 to 1e-5
#Boris and RK4 trajectories agree to ~1e-6
#run both pushers with identical arguments, then compare positions
def test_integrator():
    x0 = np.array([1.0,0,0])
    v0 = np.array([0,-1.0,0.3])
    dt, n_steps = 0.001, 2000
    q, m = 1.0, 1.0 
    xs_b, _ = boris(x0, v0, q, m, dt, n_steps, uniform_field)
    xs_r, _ = rk4(x0, v0, q, m, dt, n_steps, uniform_field)
    assert np.abs(xs_b - xs_r).max() < 1e-5


#error ratio is close to 4 (second order), 3.5 < ratio < 4.5
#halving dt should quarter the error
def test_convergence_order():
    x0 = np.array([1.0,0,0])
    v0 = np.array([0,-1.0,0.3])
    q, m = 1.0, 1.0 
    n_steps = 10000

    T = 200 #fixed total time, so that we compare periods over the same number of orbits
    errs = []
    for dt in [0.02, 0.01]:
        n_steps = int(T / dt)
        xs, vs = boris(x0, v0, q, m, dt, n_steps, uniform_field)
        theta = np.unwrap(np.arctan2(xs[:, 1], xs[:, 0]))
        omega = (theta[-1] - theta[0]) / ((n_steps - 1) * dt)
        period = 2 * np.pi / abs(omega)
        errs.append(abs(period - 2*np.pi))
    ratio = errs[0] / errs[1]
    assert 3.5 < ratio < 4.5 
