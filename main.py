import numpy as np
import matplotlib.pyplot as plt 
from analysis import divergence
from fields import uniform_field, exb_field, mirror_field, make_mirror_field, make_bottle_field, make_penning_field
from pushers import boris, rk4, euler
from scipy.signal import find_peaks
from quantum import landau_levels

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#062E92", "#8F1139", "#046C60", "#EE9E14"])


if __name__ == "__main__":
    E, B = uniform_field(np.array([1.0, 0.0, 0.0]), 0.0)
    print("E =", E)
    print("B =", B)

    dt = 0.01
    n_steps = 100000
    xs, vs = boris(
        x0=np.array([1.0, 0.0, 0.0]),
        v0=np.array([0.0, -1.0, 0.3]), #so that the gyrocenter sits at the origin and radius is 1.0
        q = 1.0,
        m = 1.0,
        dt=dt, n_steps=n_steps,
        field=uniform_field
    )
    speeds = np.linalg.norm(vs, axis=1)
    print("speed drift:", speeds.max() - speeds.min())
    #gyro-radius, all rows and just the first two columns, ignore the drift along the field 
    r = np.linalg.norm(xs[:, :2], axis=1) #perpendicular distance from z axis
    print("radius:", r.mean(), "+/-", r.std())

    #the total angle over total time gives you ω_c
    theta = np.unwrap(np.arctan2(xs[:, 1], xs[:, 0]))
    period = 2 * np.pi / abs((theta[-1] - theta[0]) / (n_steps * dt))
    print("period:", period)

    #assert: |v| constant to ~1e-14 (magnetic fields do no work, radius = 1.0, period = 2π
    #it only reads 1.0 if your gyrocenter sits at the origin
    #magnetic force is qvxB, pointing outward 

    #---------------------------------------------------------------------------------------------------------------------

    x0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, -1.0, 0.3])
    q, m = 1.0, 1.0
    dt = 0.01
    n_steps = 100000
    t= np.arange(n_steps) * dt / (2 * np.pi) #One gyroperiod is 2π, convert raw units 

    #run each integrator at two different dt, Boris and RK4 both improve as dt shrinks, but Boris's shape stays flat while RK4's stays sloped
    for dt, style in [(0.02, "--"), (0.005, "-")]:
        n_steps = int(1000 / dt)          
        t = np.arange(n_steps) * dt/(2*np.pi)
        for name, pusher in [("boris", boris), ("rk4", rk4), ("euler", euler)]:
            xs, vs = pusher(x0, v0, q, m, dt, n_steps, uniform_field)
            #vs has shape (n_steps, 3)-100,000 rows and each row a 3-component velocity vector 
            speed = np.linalg.norm(vs, axis=1) #colapse the second dim, for each row, compute √(vx² + vy² + vz²)
            #result is shape (100000,), one speed per timestep
            #fractional deviation from the starting speed, all three integrators are on the same scale
            #speed[0] is the initial speed and result is 1.0 at the start, so 0.0 at the start now
            #vertical drops in boris are log of zero, kill those off, saying below 1e-16 is indistinguishable from zero 
            rel = np.maximum(np.abs(speed / speed[0]-1.0), 1e-16) #relative speed error
            print(f"{name:6s} final |v| error: {rel[-1]:.3e}") #-1 indexing gives you the final error
            plt.plot(t, rel, style, label=f"{name} dt={dt}", linewidth=1.3, alpha=0.85)

    plt.ylim(1e-16, 1e2) #stop euler from squashing, clip the y-axis
    plt.grid(True, which='both', alpha=0.3)
    plt.title("Speed conservation, uniform B")
    plt.yscale('log') 
    plt.xlabel('time (gyroperiods)')
    plt.ylabel('relative |v| error')
    plt.legend(ncol=3, fontsize=8, loc="upper center",
    bbox_to_anchor=(0.5, -0.15), frameon=False) 
    plt.axhline(2.2e-16, color="gray", ls=":", lw=0.8) #machine epsilon, boris is at the floor of what float64 can represent 
    plt.tight_layout() 
    plt.savefig("figures/integrator_comparison.png", dpi=150)
    plt.show()

    #convergence sweep-------------------------------------------------------------------------------------------------------------
    #Boris is second-order accurate. Halve dt, the period error drops 4×. Four runs, plot error vs dt, slope should be 2
    #plot period error against dt on log-log, convergence sweep 
    #Theory says the relative period error is (ω_c·dt)²/12, so at dt = 0.01 you expect 8.3e-6. 

    #checking if rel_error and errors land on each other

    plt.figure()
    errors= [] #measured from simulations
    dt_values = [0.04, 0.02, 0.01, 0.005]
    t_total = 200 
    for dt in dt_values:
        n_steps = int(t_total / dt)
        xs, vs = boris(x0, v0, q, m, dt, n_steps, uniform_field)
        theta = np.unwrap(np.arctan2(xs[:, 1], xs[:, 0]))
        i0, i1 = 0, n_steps - 1 #loop stores xs[n] after updating
        #theta[-1] is the angle after n_steps steps. So the angle swept from theta[0] to theta[-1] corresponds to n_steps - 1
        omega = (theta[i1] - theta[i0]) / ((i1 - i0) * dt)
        period = 2 * np.pi / abs(omega)
        error = abs(period - 2*np.pi)/(2*np.pi)
        errors.append(error)
    #fit a line through the log-log points
    slope = np.polyfit(np.log(dt_values), np.log(errors), 1)[0] #gives you the slope, approximately 2.0, confirming second-order convergence
    #w_c is 1.0
    theory = (1.0 * np.array(dt_values)**2) / 12 #the error that theory predicts 
    print(f"convergence slope: {slope:.3f} (expect 2.0)")

    plt.loglog(dt_values, errors, 'o-', label="measured")
    plt.loglog(dt_values, theory, '--', label="theory: (ω_c·dt)²/12")
    plt.grid(True, which='both', alpha=0.3)
    plt.title("Boris convergence: period error vs dt")
    plt.xlabel('dt')
    plt.ylabel('period error')
    plt.legend(fontsize=8, loc="upper left", frameon=False) 
    plt.tight_layout() 
    plt.savefig("figures/period_error.png", dpi=150)
    plt.show()


    #-------------------------------------------------------------------------------------------------------------------------------------------
    # the E field pushes along x but the guiding center moves along y, perpendicular to both
    # drift direction--fit a line to xs[:, 1], magnitud (|E|/|B| = 0.5), mass and charge independence 
    # run it twice: once with q=1, m=1, once with q=-1, m=10. Different gyroradius, different rotation direction, same drift velocity
    
    plt.figure()
    dt = 0.01
    n_steps = 5000
    t = np.arange(n_steps)*dt
    x0 = np.array([0, 0, 0])
    v0 = np.array([0, -1.0, 0]) #perpendicular kick, get a cycloid, gyration + drift
    q, m = 1.0, 1.0

    #Plot xs[:, 0] against xs[:, 1] for both particles
    E_vec, B_vec = exb_field(x0, 0.0) #E is 0.15 along +x
    v_drift = np.cross(E_vec, B_vec) / np.dot(B_vec, B_vec)
    xs_c, vs_c = boris(x0, v_drift, q, m, dt, n_steps, exb_field) #
    print(f"drift-matched run, max |x| = {np.abs(xs_c[:, 0]).max():.2e} (expect ~0)")

    #independence test, drift slopes should match even though in different orbits, two particles
    for q, m in [(1, 1), (-1, 3)]:
        xs, vs = boris(x0, v0, q, m, dt, n_steps, exb_field)
        #comparing to E/B=0.5, converges over many avg orbits
        slope = np.polyfit(t, xs[:, 1], 1)[0] #fit a straight line to y versus time, gives you the slope
        print(f"q={q:+.0f} m={m:>2} drift = {slope:.4f} (expect -0.5)")
        #take all rows, first column--for x component, then all rows and the second column--for y component
        #plotting one against the other draws the path in the x-y plane, shape of the orbit, with time hidden
        plt.plot(xs[:,0], xs[:,1], label=f"q={q}, m={m}")
        plt.plot(xs[0,0], xs[0,1], 'ko', ms=4) #row 0 first timestep, column 0 x, xs[0, 1] is its y
        plt.plot(-4.5, -1, 'o', mfc='none', mec='blue', ms=12)
        plt.plot(-4.5, -1, '.', color='blue', ms=4)
        plt.text(-4.5, -1.6, 'B (out of page)', color='blue', ha='center', fontsize=8)

    plt.axhline(0, color='gray', lw=0.5)
    plt.grid(alpha=0.3)
    #plt.arrow(x, y, dx, dy), starts at origin, extends 3 units along x, none along y
    plt.arrow(0, 0, 1.2, 0, head_width=0.15, head_length=0.25, color='red', lw=1, length_includes_head=True)
    plt.text(0.6, 0.25, 'E', color='red', ha='center', fontsize=9) 
    plt.axis('equal')
    plt.xlabel('x') #since we are plotting the trajectory in the plane perpendicular to B 
    plt.ylabel('y')
    plt.title('ExB drift: same drift velocity, different orbits')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/ExB_drift.png', dpi=150)
    plt.show()


    #mirror field-----------------------------------------------------------------------------------------------------------
    #divergence check ∇·B = 0 ~1e-10, must be divergence free to keep μ (magnetic moment) conserved
    #h: classic tradeoff between truncation error (wants small h) and roundoff (wants large h)
    print(f"div B (mirror) = {divergence(mirror_field, [0.3, 0.4, 0.5]):.2e} (expect ~0)")

    #the trapped particle-----------------------------------------------------------------------------------------------------
    plt.figure()
    x0 = np.array([0.3, 0, 0]) #off-axis so it samples the radial field
    v0 = np.array([0, -0.3, 0.15]) #mostly perpendicular → should be trapped
    q, m = 1.0, 1.0
    dt = 0.01
    n_steps = 100000

    #plot xs[:, 2] (the z coordinate) against time, a clean oscillation
    # v∥ = 0.15, v⊥ = 0.3, k = 0.05 → z ≈ ±2.24

    xs, vs = boris(x0, v0, q, m, dt, n_steps, mirror_field) 
    t = np.arange(n_steps) * dt
    plt.plot(t, xs[:, 2])
    plt.axhline(2.24, color='red', ls='--', lw=0.8)
    plt.axhline(-2.24, color='red', ls='--', lw=0.8)
    plt.title("Trapped particle")
    plt.xlabel('time')
    plt.ylabel('z')
    plt.tight_layout() 
    plt.savefig("figures/trapped_particle.png", dpi=150)
    plt.show()

    #Plot mu against t--------------------------------------------------------------------------------------------------------
    plt.figure()
    Bm = np.array([np.linalg.norm(mirror_field(x, 0.0)[1]) for x in xs])
    Bhat = np.array([mirror_field(x, 0.0)[1] / np.linalg.norm(mirror_field(x, 0.0)[1]) for x in xs])
    vpar = np.sum(vs * Bhat, axis=1)
    vperp2 = np.sum(vs**2, axis=1) - vpar**2
    mu = vperp2 / (2 * Bm)
    print(f"mu: {mu[0]:.8f} → {mu[-1]:.5f}")
    plt.plot(t, mu)
    plt.title('Magnetic moment μ = mv⊥²/2B is conserved')
    plt.xlabel('time (t)')
    plt.ylabel('mu (μ)')
    plt.tight_layout() 
    plt.savefig("figures/mu_conservation.png", dpi=150)
    plt.show()

    #field varying (contrast)----------------------------------------------------------------------------------------------------------------------------------------
    #At k=0.8 the trap tightens to ±0.56 while the gyroradius stays 0.3, the two scales become comparable and the invariant should visibly degrade
    plt.figure()
    for k in [0.05, 0.8]: #now the field varies over a scale comparable to the gyroradius, the adiabatic condition fails, and μ should visibly wander
        fld = make_mirror_field(k=k)
        xs, vs = boris(x0, v0, q, m, dt, n_steps, fld)
        Bm = np.array([np.linalg.norm(fld(x, 0.0)[1]) for x in xs])
        Bhat = np.array([fld(x, 0.0)[1] / np.linalg.norm(fld(x, 0.0)[1]) for x in xs])
        vpar = np.sum(vs * Bhat, axis=1) #projects velocity onto the field direction (dot product with the unit vector)
        vperp2 = np.sum(vs**2, axis=1) - vpar**2 #the rest by Pythagoras
        mu = vperp2 / (2 * Bm)
        plt.plot(t, mu/mu[0], label=f"k={k}") #y label is the normalized mu, to see the drift
    plt.axhline(1.0, color='gray', lw=0.5)
    plt.title("Adiabatic invariance breaks when scales overlap")
    plt.xlabel('time (t)')
    plt.ylabel('μ / μ₀')
    plt.grid(alpha=0.3)
    plt.tight_layout() 
    plt.legend()
    plt.savefig("figures/mu_adiabatic_breakdown.png", dpi=150)
    plt.show()

    #loss cone ----------------------------------------------------------------------------------------------------------------------------
    #1.39e-12, bottle field is divergence-free, B_z peaks at z = a
    #Mirror ratio R = 2.766, so sin²θ_loss = 1/2.766 = 0.362 → θ_loss ≈ 36.8°.
    
    a = 4.0
    fld = make_bottle_field(k=0.3, a=a)
    print(f"div B = {divergence(fld, [0.3, 0.4, 2.0]):.2e} (expect ~0)")
    B_max = np.linalg.norm(fld([0,0,a], 0.0)[1])
    R = B_max
    theta_loss = np.rad2deg(np.arcsin(np.sqrt(1/R)))
    print(f"R = {R:.3f}, theta_loss = {theta_loss:.1f}°")
    #pick z max so that B_z doesnt grow indefinetly
    plt.figure()
    z_max = 4.0
    q, m = 1.0, 1.0
    n_steps = 20000
    t = np.arange(n_steps) * dt
    thetas = np.array([60, 45, 25])
    for theta in thetas:
        v_perp = 0.4 * np.sin(np.deg2rad(theta))
        v_parl = 0.4 * np.cos(np.deg2rad(theta))
        v0 = np.array([0, -v_perp, v_parl])
        x0 = np.array([v_perp/1.0, 0, 0])
        xs, vs = boris(x0, v0, q, m, dt, n_steps, fld)
        print(f"θ={theta}°  trapped={np.sin(np.deg2rad(theta))**2 > 1/R}  max|z|={np.abs(xs[:,2]).max():.2f}")
        plt.plot(t, xs[:, 2], label=f"θ={theta:.0f}°")
    plt.axhline(z_max, color='gray', lw=0.5, ls="--", label=f"throat, θ_loss={theta_loss:.1f}°")
    plt.text(150, 4.3, 'throat (B max)', color='gray', fontsize=8)
    plt.axhline(-z_max, color='gray', lw=0.5)
    plt.ylim(-6, 15)
    plt.title("Loss cone: same speed, different pitch angles")
    plt.xlabel('time (t)')
    plt.ylabel('z')
    plt.grid(alpha=0.3)
    plt.tight_layout() 
    plt.legend()
    plt.savefig("figures/loss_cone.png", dpi=150)
    plt.show()


    #Penning Trap-----------------------------------------------------------------------------------------------------------------------------------
    x0 = np.array([1.0, 0, 0.5]) #starting point
    v0 = np.array([0, 0.1, 0]) #small velocity
    n_steps = 200000
    C = 0.05
    q, m = 1.0, 1.0
    B0 = 1.0
    fld = make_penning_field(B0=1.0, C=C)
    print(f"Div E = {divergence(fld, [0.3, 0.4, 0.5], which=0):.2e} (expect ~0)")
    w_c = (q * B0) / m
    w_z = np.sqrt(2*q*C/m) #axial oscillation 
    disc = np.sqrt(w_c**2/4 - w_z**2/2)
    w_c_prime = w_c/2 + disc #modified cyclotron
    w_m = w_c/2 - disc #magneton
    print(f"w_c={w_c:.4f} w_z={w_z:.4f} w'_c={w_c_prime:.4f} ω_m={w_m:.4f}")
    #the two verifications
    print(f"w'_c + w_m = {w_c_prime + w_m:.6f}  (expect {w_c:.6f})")
    print(f"ω'_c · ω_m = {w_c_prime*w_m:.6f}  (expect {w_z**2/2:.6f})")
    #the trajectory 
    dt = 0.01
    t = np.arange(n_steps)*dt
    xs, vs = boris(x0, v0, q, m, dt, n_steps, fld)

    print(np.abs(xs).max()) #should stay around 1-2
    plt.figure()
    plt.plot(xs[:, 0], xs[:, 1]) #the rosette in the radial plane, spatial a and y
    plt.axis('equal')  #equal scaling 
    plt.title("Penning trap: radial orbit (cyclotron + magnetron)")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/penning_rosette.png", dpi=150)
    plt.show()
    #ω_z = 0.316, period 2π/0.316 = 19.9, and the peaks sit at 20 and 40

    plt.figure()
    plt.plot(t[:5000], xs[:5000, 2]) #the axial osciallation, a clean sine at w_z, temporal z and t 
    plt.title(f"Penning trap: axial oscillation, ω_z = {w_z:.3f}")
    plt.xlabel('time')
    plt.ylabel('z')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/penning_axial.png", dpi=150)
    plt.show()

    #The FFT, confirm the orbit is bounded 
    #two spectra--radial and axial
    freqs = np.fft.rfftfreq(n_steps, dt) * 2*np.pi
    spec_x = np.abs(np.fft.rfft(xs[:, 0])) #rfftfreq gives ordinary frequencies in cycles per unit time
    #multiplying by 2π converts to angular frequency so it's comparable to the ω values
    spec_z  =np.abs(np.fft.rfft(xs[:, 2]))
    plt.figure()
    plt.plot(freqs, spec_x, label="x")
    plt.plot(freqs, spec_z, label="z")
    plt.xlim(0, 1.2) 
    for w, name in [(w_c_prime, "ω'_c"), (w_z, "ω_z"), (w_m, "ω_m")]:
        plt.axvline(w, color='gray', ls='--', lw=0.8) #vertical lines at the three predicted values 
        plt.text(w, spec_x.max()*1.3, name, fontsize=8, ha='center', color='gray')
    #The x-spectrum should show peaks at ω'_c and ω_m; the z-spectrum a single peak at ω_z
    plt.yscale('log')
    plt.ylim(spec_x.max()*1e-5, spec_x.max()*5) 
    plt.title("Penning trap: mode (radial and axial) spectrum")
    plt.xlabel("angular frequency ω")
    plt.ylabel("|FFT(x and z)|")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.savefig('figures/penning_spectrum.png', dpi=150)
    plt.show()

    idx, _ = find_peaks(spec_x, height=spec_x.max()*0.01)
    idz, _ = find_peaks(spec_z, height=spec_z.max()*0.01)
    print("measured radial peaks:", freqs[idx])
    print(f"ω_z : measured {freqs[idz][0]:.4f}  predicted {w_z:.4f}")
    print(f"ω'_c: measured {freqs[idx].max():.4f}  predicted {w_c_prime:.4f}")
    print(f"ω_m : measured {freqs[idx].min():.4f}  predicted {w_m:.4f}")
    #measured vs predicted
    #ω_z = 0.316, disc = √(0.25 - 0.05) = 0.4472, giving ω'_c = 0.9472 and ω_m = 0.0528
    #measured 0.9488 vs 0.9472, and 0.0534 vs 0.0528, agreement to ~0.2%, limited by FFT bin width (2π/(n_steps·dt) = 0.0031)
    #three frequencies extracted from a simulation, matching three numbers derived from a quadratic

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #quantum landau levels convergence, just like Boris convergence sweep 
    N_values = np.array([500, 1000, 2000, 4000])
    errors= []
    dxs = []
    plt.figure()
    for n in N_values:
        x, evals, _ = landau_levels(N=n)
        dxs.append(20.0 / (n-1))
        errors.append(abs(evals[5]-5.5))
    slope = np.polyfit(np.log(dxs), np.log(errors), 1)[0] 
    plt.loglog(dxs, errors, 'o-', label="measured")
    print(f"convergence slope: {slope:.3f} (expect 2.0)")
    plt.grid(True, which='both', alpha=0.3)
    plt.title("Landau level convergence: n=5 eigenvalue")
    plt.xlabel('Δx')
    plt.ylabel('|E₅ − 5.5|')
    plt.legend(fontsize=8, loc="upper left", frameon=False) 
    plt.tight_layout() 
    plt.savefig("figures/landau_convergence.png", dpi=150)
    plt.show()


    #degeneracy--------------------------------------------------------------------------------------------------------------------
    #E_n doesn't depend on k_y, which enters only through x₀ = -ħk_y/(qB₀). 
    #Different k_y means the oscillator is centered at a different x, but the energy is identical
    #Infinitely many states (one per k_y) at the same energy, which is the foundation of the quantum Hall effect
    #all three must be ~1e-6
    x0s = np.array([0.0, 2.0, -3.0])
    ref = None 
    for x0_val in x0s: #x: the center of the oscillation
        _, evals, _ = landau_levels(x0_val)
        if ref is None:
            ref = evals
        #should be ~1e-15 for identical
        #energy doesnt depend on where the oscillator is centered 
        print(f"x0={x0_val:.1f} max deviation from x0=0: {np.abs(evals-ref).max():.2e}")


    #wave functions-----------------------------------------------------------------------------------------------
    #harmonic oscillator eigenfunctions: a Gaussian, then one node, two nodes, three nodes
    plt.figure()
    x, evals, evecs = landau_levels()
    for n in range(4):
        #adding evals[n] offsets each curve vertically to its own energy level
        #eigh_tridiagonal normalizes eigenvectors to unit L2 norm over the grid
        #so scale them *3
        plt.plot(x, evecs[:, n]*3 + evals[n], label=f"n={n}")
    plt.plot(x, 0.5*x**2, 'k--', label="V(x) = ½x²") #potential
    plt.ylim(0, 6)
    plt.xlim(-5, 5)
    plt.title('Landau eigenstates in the Landau gauge')
    plt.xlabel('x')
    plt.ylabel('energy / ψ')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/landau_states.png', dpi=150)
    plt.show()
    #Gaussian at 0.5, one node at 1.5, two nodes at 2.5, three at 3.5 
    #each sitting inside the parabola, with the wavefunction dying where the parabola crosses its energy


    #the bridge figure---------------------------------------------------------------------------------------------------------
    #plot classical energy against radius as a smooth parabola, then mark the allowed r_n as discrete points on it

    #Classical: a dense r array from 0 to 4, energy E = 0.5 * r**2 (m = ω_c = 1). Plot as a continuous line, any radius is allowed.
    #Plot quantum as markers ('o') on top of the classical curve.
    #the points should be exactly on the parabola, 0.5·(2n+1) = n + 0.5
    plt.figure()
    r = np.linspace(0, 3.5, 500)
    E = 0.5 * r**2
    n = np.arange(6)
    r_n = np.sqrt(2*n + 1) 
    E_n = n + 0.5 
    #x-axis is the radius and y-axis is the energies
    #both plots take a two-element x list and a two-element y list
    plt.plot(r, E, label='classical: any radius allowed')

    for i in range(6):
        plt.plot([0, r_n[i]], [E_n[i], E_n[i]], "k:", lw=0.6) #E_n
        plt.plot([r_n[i], r_n[i]], [0, E_n[i]], "k:", lw=0.6) #r_n
    
    plt.plot(r_n, E_n, 'o', ms=8, label='quantum: r_n = l_B√(2n+1)')
    plt.title('Bohr–Sommerfeld: the classical orbit becomes a Landau level')
    plt.xlabel('cyclotron radius r / l_B')
    plt.ylabel('energy / ħω_c')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/bridge_classical_quantum.png', dpi=150)
    plt.show()

'''
classical:  E = ½mω_c²r²
quantum:    E_n = ħω_c(n + ½)
equate:     r_n = √(ħ(2n+1)/mω_c) = l_B√(2n+1)

One figure, three kinds of plt.plot call drawn onto it:
    The parabola, the classical continuum
    Twelve short dotted segments (six horizontal, six vertical), the guides
    Six markers, the quantum states
'''