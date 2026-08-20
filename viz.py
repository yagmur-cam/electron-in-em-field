#Pyvista scenes 
import pyvista as pv
import numpy as np 
from fields import make_penning_field, uniform_field, exb_field, mirror_field, make_bottle_field
from pushers import boris

# xs is (n_steps, 3)
#Penning trajectory-----------------------------------------------------------------------------------------------------------
#xs[:50000:10], 200,000 points makes a tube with millions of triangles and it'll crawl

def tube_from_points(pts, radius=0.02, scalars=None, name="speed"):
    n = len(pts)
    poly = pv.PolyData(pts)
    poly.lines = np.hstack([[n], np.arange(n)])
    if scalars is not None:
        poly[name] = scalars
    return poly.tube(radius=radius)

if __name__ == "__main__":
    q, m = 1.0, 1.0
    x0 = np.array([1.0, 0, 0.5])
    v0 = np.array([0, 0.1, 0])
    n_steps = 200000
    dt = 0.01
    fld = make_penning_field(B0=1.0, C=0.05)
    xs, vs = boris(x0, v0, q, m, dt, n_steps, fld)

    step = 3 #xs and vs both use this stride 
    sub = xs[::step] #subsample, 20,000 points
    speeds = np.linalg.norm(vs[::step], axis=1) #Attach scalars to poly before calling .tube()
    tube = tube_from_points(sub, radius=0.015, scalars=speeds)
    p = pv.Plotter()

    p.show_grid(color="white", font_size=10,
                bounds=[-1.5, 1.5, -1.5, 1.5, -1.5, 1.5])

    plane = pv.Plane(center=(0, 0, xs[:,2].min()), direction=(0,0,1),
                i_size=6, j_size=6, i_resolution=12, j_resolution=12)
    p.add_mesh(plane, style="wireframe", color="#2a4a6a", opacity=0.5)
    arrow = pv.Arrow(start=(2.5,0,-1), direction=(0,0,1), scale=2)
    p.add_mesh(arrow, color="#5DCAA5")
    p.add_point_labels([(2.5,0,1.7)], ["B"], text_color="white", font_size=14,
                        shape=None, show_points=False)
    
    pv.global_theme.font.color = "white"
    p.add_text("Penning trap: three-mode orbit",
               position="upper_left", font_size=12, color="white")
    p.set_background("black", top="#101828")
    p.add_mesh(tube, scalars="speed", cmap="plasma", 
               smooth_shading=True, specular=0.5,
               scalar_bar_args={
                    "color": "white", "title": "speed\n",
                    "vertical": True,
                    "height": 0.5, "width": 0.05, 
                    "position_x": 0.88,
                    "position_y": 0.25, "title_font_size": 14,
                    "label_font_size": 12,
               })
    p.camera_position = [(12, 10, 6), (0, 0, 0), (0, 0, 1)]
    p.show(auto_close=False)
    p.screenshot("figures/penning_3d.png")
    p.close()


#uniform B helix---------------------------------------------------------------------------------------------------------
    #speed is constant, magnetic field does no work
    q, m = 1.0, 1.0
    x0 = np.array([1.0, 0, 0])
    v0 = np.array([0, -1, 0.3])
    dt = 0.01
    n_steps = 3000
    xs, vs = boris(x0, v0, q, m, dt, n_steps, uniform_field)
    speeds = np.linalg.norm(vs, axis=1) 
    tube = tube_from_points(xs, radius=0.04)
    p = pv.Plotter()

    p.show_grid(color="white", font_size=10,
                bounds=[-2, 2, -2.5, 2.5, -1, 9])

    plane = pv.Plane(center=(0, 0, xs[:,2].min()), direction=(0,0,1),
                i_size=6, j_size=6, i_resolution=12, j_resolution=12)
    p.add_mesh(plane, style="wireframe", color="#2a4a6a", opacity=0.5)
    arrow = pv.Arrow(start=(0,0,-1), direction=(0,0,1), scale=3)
    p.add_mesh(arrow, color="#5DCAA5")
    p.add_point_labels([(0,0,3.2)], ["B"], text_color="white", font_size=14,
                       shape=None, show_points=False)
    
    p.set_background("black", top="#101828")
    p.add_text("Uniform B: helical orbit",
               position="upper_left", font_size=12, color="white")
    p.add_mesh(tube, color="orange", 
                smooth_shading=True, specular=0.5,)
    p.camera_position = [(-9, 7, 4), (0, 0, 4), (0, 0, 1)]
    p.camera.zoom(0.5)
    p.show(auto_close=False)
    p.screenshot("figures/uniform_3d.png")
    p.close()
    

#ExB cycloid, the drift---------------------------------------------------------------------------------------------
    #spiral along z while drifting along y, two helices of different radius translation together, 3D
    #set up p → loop (boris, tube, add_mesh) → grid, plane, arrows, labels → background → show
    x0 = np.array([0, 0, 0])
    v0 = np.array([0, -1, 0.2]) 
    dt = 0.01
    n_steps = 4000
    p = pv.Plotter() 
    for (q, m), col in [((1, 1), "#1E63DB"), ((-1, 3), "#C41244")]:
        xs, vs = boris(x0, v0, q, m, dt, n_steps, exb_field)
        speeds = np.linalg.norm(vs, axis=1) 
        tube = tube_from_points(xs, radius=0.04)
        p.add_mesh(tube, color=col, 
            smooth_shading=True, specular=0.5,)

    p.add_text("blue: q=+1, m=1\nred: q=-1, m=3", position="upper_left",
               font_size=8, color="white")
    p.show_grid(color="white", font_size=10, bounds=[-3, 3, -9, 1, -1, 9])
    span = 2 * max(np.abs(xs[:, :2]).max(), 1.0)
    plane = pv.Plane(center=(0, xs[:,1].mean(), xs[:,2].min()), 
                direction=(0,0,1), i_size=span, j_size=span, 
                i_resolution=12, j_resolution=12)
    p.add_mesh(plane, style="wireframe", color="#2a4a6a", opacity=0.5)
    arrowB = pv.Arrow(start=(0,0,-1), direction=(0,0,1), scale=2)
    arrowE = pv.Arrow(start=(0,0,0), direction=(1,0,0), scale=2)
    p.add_mesh(arrowB, color="#5DCAA5")
    p.add_mesh(arrowE, color="#E24B4A")
    p.add_point_labels([(0,0,1.3)], ["B"], text_color="white", font_size=14,
                        shape=None, show_points=False)
    p.add_point_labels([(2.3,0,0)], ["E"], text_color="white", font_size=14,
                            shape=None, show_points=False)
        
    p.set_background("black", top="#101828")
    p.add_text("E×B drift: same drift velocity, different orbits",
               position="upper_right", font_size=12, color="white")
    p.camera_position = [(9, 6, 10), (0, -4, 4), (0, 0, 1)]
    p.camera.zoom(0.4)
    p.show(auto_close=False)
    p.screenshot("figures/exb_drift_3d.png")
    p.close()


#magnetic mirror------------------------------------------------------------------------------------------------------------
    #field lines that are nearly straight in the middle and pinch inward toward z = ±4, 
    #with the particle spiraling between them and turning around before the pinch
    #use the point positions and the B vectors 
    #trace lines that follow the vector "streamlines"

    #The coil is dense and tight at the turning points, 
    #loose and stretched through the middle: v∥ going to zero at the ends and peaking at the midplane. 
    q, m = 1.0, 1.0 
    x0 = np.array([0.3, 0, 0])
    v0 = np.array([0, -0.35, 0.42])  #made the velocity more parallel  
    dt = 0.01
    n_steps = 12000 #the bounce period grows as the particle travels further
    fld = make_bottle_field(k=0.3, a=4.0)
    xs, vs = boris(x0, v0, q, m, dt, n_steps, fld)

    xg = np.linspace(-2, 2, 20)
    yg = np.linspace(-2, 2, 20)
    zg = np.linspace(-4, 4, 20)
    #X[i,j,k], Y[i,j,k], Z[i,j,k]--one points coordinates 
    X, Y, Z =  np.meshgrid(xg, yg, zg, indexing='ij')
    #ask your field for B at each of those points, collect the calls result in an array 
    Bs = []
    #X.ravel() turns the 20×20×20 array into a flat list of 8000 numbers
    #pts is (8000, 3)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    p = pv.Plotter()
    for pt in pts: #each iteration gives you one point's (x, y, z)
        B = fld(pt, 0.0)[1]
        Bs.append(B)
    Bs = np.array(Bs)
    print(np.shape(Bs))
    print("max|z| :", np.abs(xs[:,2]).max()) 
    grid = pv.StructuredGrid(X, Y, Z)
    grid["B"] = Bs
    streams = grid.streamlines(
        "B",
        source_radius=0.5, source_center=(0, 0, -3.5), #define a sphere where the streamlines start
        n_points=40, max_length=15, 
        integration_direction="both", #traces forward and backward from each seed so the lines run the full length of the trap
    )
    p.add_text("Magnetic mirror: trapped particle and field lines",
               position="upper_left", font_size=12, color="white")
    p.add_mesh(streams.tube(radius=0.015), color="#5DCAA5", opacity=0.5)
    tube = tube_from_points(xs, radius=0.03)
    p.add_mesh(tube, color="#EE9E14", smooth_shading=True, specular=0.5)
    p.show_grid(color="white", font_size=10)
    p.set_background("black", top="#101828")
    p.add_axes(color="white")
    p.camera_position = [(-10, 0, 0), (0, 0, 0), (0, 0, 1)]
    p.camera.zoom(0.7)
    p.show(auto_close=False)
    p.screenshot("figures/magnetic_mirror_3d.png")
    p.close()


#loss cone, escaping particles----------------------------------------------------------------------------------------------------------
    #two trajectories in the same bottle field, one trapped and one escaping, both tubed, different colors, same streamlines
    q, m = 1.0, 1.0 
    dt = 0.01
    n_steps = 10000 
    fld = make_bottle_field(k=0.3, a=4.0)

    xg = np.linspace(-2, 2, 20)
    yg = np.linspace(-2, 2, 20)
    zg = np.linspace(-8, 8, 50)
    X, Y, Z =  np.meshgrid(xg, yg, zg, indexing='ij')
    p = pv.Plotter()
    for theta, col in [(45, "#EE9E14"), (25, "#C41244")]:
        th = np.deg2rad(theta)
        v0 = np.array([0, -0.4*np.sin(th), 0.4*np.cos(th)])
        x0 = np.array([0.4*np.sin(th), 0, 0]) #built from v_perp so the gyrocenter sits on the axis
        xs, vs = boris(x0, v0, q, m, dt, n_steps, fld)
        keep = np.abs(xs[:, 2]) < 7
        xs = xs[keep]
        print(f"θ={theta}°  max|z| = {np.abs(xs[:,2]).max():.2f}")
        tube = tube_from_points(xs, radius=0.03)
        p.add_mesh(tube, color=col, smooth_shading=True, specular=0.5)

    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    Bs = np.array([fld(pt, 0.0)[1] for pt in pts])
    grid = pv.StructuredGrid(X, Y, Z)
    grid["B"] = Bs
    streams = grid.streamlines(
        "B",
        source_radius=1.5, source_center=(0, 0, -3.5), 
        n_points=40, max_length=8, 
        integration_direction="both", terminal_speed=1e-8,
    )
    tubes = streams.tube(radius=0.015).clip_box([-2, 2, -2, 2, -4, 5], invert=False)
    p.add_mesh(tubes, color="#5DCAA5", opacity=0.5)
    p.add_text("Loss cone: escaping particle from magnetic bottle",
               position="upper_left", font_size=12, color="white")
    p.add_text("orange: θ=45° (trapped)\nred: θ=25° (escapes)",
               position="lower_left", font_size=10, color="white")
    p.show_grid(color="white", font_size=10, bounds=[-1.5, 1.5, -1.5, 1.5, -4, 5])
    p.set_background("black", top="#101828")
    p.add_axes(color="white")
    p.camera_position = [(-16, 2, 1), (0, 0, 1), (0, 0, 1)]
    p.camera.zoom(0.7)
    p.camera.up = (0, 0, 1)
    p.show(auto_close=False)
    p.screenshot("figures/loss_cone_3d.png")
    p.close()


#-------------------------------------------------------------------------------------------------------------------------
#A classical helix, orange tube, uniform B, several turns
#Six horizontal rings at radii √(2n+1) = 1.00, 1.73, 2.24, 2.65, 3.00, 3.32, centered on the field axis
#A B arrow along z
#The rings are the quantized orbits. The helix passes through, or near, one of them, showing classical trajectory is one allowed member of a discrete family
#create plotter → run boris → tube the helix → add it → loop drawing rings and labels → arrow, background, camera, title → show/screenshot
    p = pv.Plotter()
    #F = qv×B
    x0 = np.array([1.732, 0, 0]) #gyrocenter sits at the origin and the helix lands exactly on the n=1 ring
    v0 = np.array([0, -1.732, 0.1])
    q, m = 1.0, 1.0
    dt = 0.01
    n_steps = 1256 #exactly two periods 
    #12 time units, and the gyroperiod is 2π ≈ 6.28, 1.9 turns
    xs, vs = boris(x0, v0, q, m, dt, n_steps, uniform_field) #gyroradius is √3
    print(xs[:,0].mean(), xs[:,1].mean())
    tube = tube_from_points(xs, radius=0.03)
    p.add_mesh(tube, color="#EE9E14", smooth_shading=True, specular=0.5)
    arrow = pv.Arrow(start=(0,-3,0), direction=(0,0,1), scale=1.5)
    p.add_mesh(arrow, color="#5DCAA5")
    p.add_point_labels([(0,-3,1.8)], ["B"], text_color="white", font_size=14,
                shape=None, show_points=False)
    for n in range(6):
        r = np.sqrt(2*n + 1) 
        t = np.linspace(0, 2*np.pi, 200) 
        pts = np.column_stack([r*np.cos(t), r*np.sin(t), np.zeros_like(t)])
        ring = tube_from_points(pts, radius=0.02)
        p.add_mesh(ring, color="#7F77DD", opacity=0.8)
        ang = n * np.pi / 3
        p.add_point_labels([(r*np.cos(ang), r*np.sin(ang), 0)], [f"n={n}"], text_color="white",
                        font_size=10, shape=None, show_points=False)
    p.add_text("Classical orbit as a Landau level: r_n = l_B√(2n+1)",
        position="upper_left", font_size=12, color="white")
    p.show_grid(color="white", font_size=10, bounds=[-3.5, 3.5, -3.5, 3.5, -1, 6])
    p.set_background("black", top="#101828")
    p.add_axes(color="white")
    p.camera_position = [(8, 8, 7), (0, 0, 1), (0, 0, 1)]
    p.camera.zoom(0.7)
    p.camera.up = (0, 0, 1)
    p.show(auto_close=False)
    p.screenshot("figures/bridge_figure_3d.png")
    p.close()



#--------------------------------------------------------------------------------------------------------------------------------
#Penning Trap animation
#render the tube frame by frame, each frame showing one more chunk of the trajectory, so the orbit draws itself
    q, m = 1.0, 1.0
    x0 = np.array([1.0, 0, 0.5])
    v0 = np.array([0, 0.1, 0])
    n_steps = 200000
    dt = 0.01
    fld = make_penning_field(B0=1.0, C=0.05)
    xs, vs = boris(x0, v0, q, m, dt, n_steps, fld)

    sub = xs[::20] 
    p = pv.Plotter(off_screen=True)

    #p.show_grid(color="white", font_size=10,
                #bounds=[-1.5, 1.5, -1.5, 1.5, -1.5, 1.5])
    plane = pv.Plane(center=(0, 0, xs[:,2].min()), direction=(0,0,1),
                i_size=6, j_size=6, i_resolution=12, j_resolution=12)
    p.add_mesh(plane, style="wireframe", color="#2a4a6a", opacity=0.5)
    arrow = pv.Arrow(start=(2.5,0,-1), direction=(0,0,1), scale=2)
    p.add_mesh(arrow, color="#5DCAA5")
    p.add_point_labels([(2.5,0,1.7)], ["B"], text_color="white", font_size=14,
                        shape=None, show_points=False)
    
    pv.global_theme.font.color = "white"
    p.add_text("Penning trap: three-mode orbit",
               position=(20, 950), font_size=12, color="white")
    p.set_background("black", top="#101828")
    p.camera_position = [(12, 10, 6), (0, 0, 0), (0, 0, 1)]

    p.open_gif("figures/penning_anim.gif", fps=20)
    p.add_axes(color="white")
    p.show(auto_close=False)       

    step_i = 10
    frames = len(range(10, len(sub), step_i))
    dtheta = 180/frames #half revolution
    for i in range(20, len(sub), step_i):
        partial = tube_from_points(sub[:i], radius=0.015)
        p.add_mesh(partial, color="#EE9E14", name="traj",
                   smooth_shading=True)
        head = pv.Sphere(radius=0.05, center=sub[i-1]) #moving sphere 
        p.add_mesh(head, color="white", name="head", render_points_as_spheres=True)
        p.camera.azimuth += dtheta
        p.write_frame()

    p.close()