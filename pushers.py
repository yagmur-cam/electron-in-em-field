#Boris, RK4, Euler
import numpy as np

#returns xs, vs, arrays of shape (n_steps, 3)
#velocity lives at half-steps, so back up v0
#Advances the position and velocity of a charged particle by one time step
#uniform B, no E, run 10**5 steps, speed must remain constant 
#dt = 0.01, v0 = [0, 1, 0.3]
#q = -1, m = 1 for an electron and q = 1, m = 1836 for a proton, ExB drift is mass independent 
def boris(x0, v0, q, m, dt, n_steps, field): 
    xs = np.zeros((n_steps, 3)) #storage arrays
    vs = np.zeros((n_steps, 3))
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)
    E, B = field(x, 0.0) 
    #half-step initialization
    #Boris expects v at t = -dt/2, push v0 backward half a step to not have the trajectory being offset from the start 
    v = v - (q * dt / (2.0 * m)) * (E + np.cross(v, B))

    alpha = q * dt / (2.0 * m)
    for n in range(n_steps):
        t_now = n * dt 
        E, B = field(x, t_now)
        #the first-step of electric acc
        #v0 = v**n-1/2
        v_minus = v + alpha * E 
        #magnetic field rotation, vector t parameterizes the rotation angle
        t = alpha * B #the rotation vector
        t_mag = np.dot(t, t) 
        #vector s ensures that the scheme remains second-oder accurate
        s = (2.0 * t) / (1.0 + t_mag)
        #we perform the rotation in two sub-steps
        v_prime = v_minus + np.cross(v_minus, t) 
        v_plus = v_minus + np.cross(v_prime, s)
        #second half-step of the elctric field acc
        v_next = v_plus + alpha * E 
        #update position using the new midpoint velocity
        #x0 is x**n
        x_next = x + dt * v_next

        x= x_next
        v = v_next

        xs[n] = x
        vs[n] = v #vs[n] holds the half-step velocity which is offset by dt/2 from xs[n]
        #for |v| conservation check this doesnt matter but for the energy plot it matters
    return xs, vs #shape (n_steps, 3)


#works on a general first-order system dy/dt = f(y, t)
#state is 6-dimensional, position and velocity 
#y = [x, v] (6 components)
#f(y, t) = [v, (q/m)(E + v × B)]
#4 calls 
def rk4(x0, v0, q, m, dt, n_steps, field):
    xs = np.zeros((n_steps, 3))
    vs = np.zeros((n_steps, 3))
    #only rk4 needs the stacked 6-vector form 
    y = np.concatenate([np.array(x0, dtype=float),
                        np.array(v0, dtype=float)])
    for n in range(n_steps):
        t_now = n * dt
        k1 = rk4_helper(y, t_now, q, m, field)
        k2 = rk4_helper(y+(dt/2)*k1, t_now+(dt/2), q, m, field)
        k3 = rk4_helper(y+(dt/2)*k2, t_now+(dt/2), q, m, field)
        k4 = rk4_helper(y+(dt*k3), t_now+dt, q, m, field)
        y = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        xs[n] = y[:3]
        vs[n] = y[3:]
    return xs, vs

#takes the 6-vector and returns the 6-vector derivative
#compute time-derivative of state y, y = [x, v], return [dx/dt, dv/dt]
#dx/dt is just v, dv/dt is acceleration, (q/m)(E + v × B), with E, B from field(x, t)
def rk4_helper(y, t, q, m, field): #y is the 6-vector state 
    x = y[:3]
    v = y[3:]
    dxdt = v
    E, B = field(x, t) 
    dvdt = (q/m) * (E + np.cross(v, B))
    return np.concatenate([dxdt, dvdt])


def euler(x0, v0, q, m, dt, n_steps, field):
    xs = np.zeros((n_steps, 3)) #
    vs = np.zeros((n_steps, 3))
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)
    for n in range(n_steps):
        t_now = n * dt 
        E, B = field(x, t_now)
        v_next = v + dt * (q/m) * (E + np.cross(v, B))
        x_next = x + dt * v 
        x= x_next
        v = v_next
        xs[n] = x
        vs[n] = v
    return xs, vs 


#plot |v| against time on a log y-axis. Euler grows exponentially, RK4 decays steadily, Boris is a flat line at machine precision
#x_next = x + dt * v uses the old velocity. That's forward Euler, and it blows up fastest, Using v_next instead gives semi-implicit Euler, noticeably better behaved

