import numpy as np 

#q = m = 1,  B0 = 1  →  ω_c = 1
#time is in gyroperiods, length is in Larmor radius 

#every field addec takes (r, t) and returns two arrays of shape (3,)
def uniform_field(r, t): # ---> (E, B), both has shape (3,)
    E = np.zeros(3) #x, y, z axis 
    B = np.array([0.0, 0.0, 1.0])
    return E, B 


#ExB drift
def exb_field(r, t): # E is along x and B is along z 
    E = np.array([0.15, 0.0, 0.0]) 
    B = np.array([0.0, 0.0, 1.0])
    return E, B 


def mirror_field(r, t):
    x, y, z = r
    B0 = 1.0
    k = 0.05
    B_x = -B0 * k * x *z
    B_y = -B0 * k * y *z
    B_z = B0 * (1 + k * z**2)
    E = np.zeros(3)
    B = np.array([B_x, B_y, B_z])
    return E, B 


def make_mirror_field(B0=1.0, k=0.05):
    def field(r, t):
        x, y, z = r
        B_x = -B0 * k * x *z
        B_y = -B0 * k * y *z
        B_z = B0 * (1 + k * z**2)
        E = np.zeros(3)
        B = np.array([B_x, B_y, B_z])
        return E, B 
    return field


def make_bottle_field(B0=1.0, k=0.3, a=4.0):
    def field(r, t):
        x, y, z = r
        # B_z = B0 (1 + k·z²·exp(-z²/z_max²))
        # B_r = -B0·k·r·z·exp(-z²/a²)·(1 - z²/a²)
        common = B0 * k * np.exp(-z**2 / a**2)
        B_x = - common * x * z * (1 - z**2 / a**2)
        B_y = -common * y * z * (1 - z**2 / a**2)
        B_z = B0 * (1 + k * z**2 * np.exp(-z**2 / a**2))
        E = np.zeros(3)
        B = np.array([B_x, B_y, B_z])
        return E, B 
    return field 


def make_penning_field(B0=1.0, C=0.05): #C = V0 / 2d**2, control knob
    def field(r, t):
        x, y, z = r
        E = np.array([C*x, C*y, -2*C*z])
        B = np.array([0.0, 0.0, B0])
        return E, B
    return field 
