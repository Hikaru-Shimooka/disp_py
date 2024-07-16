# parameter settings

# mass
mi = 1  # ion
mr = 1/1836  # mass ratio
me = mi*mr  # electron

# temperature
Ti = 1  # ion
Tec = Ti  # cool electron
# Teh = Tec  # hot electron
# Teh = 10*Tec  # hot electron
Teh = 100*Tec  # hot electron

# plasma frequency
wpi = 1  # ion
wpec = wpi*((0.5*mi/me)**(1/2))  # cool electron
wpeh = wpi*((0.5*mi/me)**(1/2))  # hot electron

# debye wave number
ki = wpi*((mi/Ti)**(1/2))  # ion
kec = wpec*((me/Tec)**(1/2))  # cool electron
keh = wpeh*((me/Teh)**(1/2))  # hot electron

# thermal velocity
vec = (Tec/me)**(1/2)  # cool electron
veh = (Teh/me)**(1/2)  # hot electron
vi = (Ti/mi)**(1/2)  # ion

# k range
k_min, k_max, k_num = 0, 0.5, 200

# wr range
wr_min, wr_max, wr_num = 0, 80, 400

# wi range
wi_min, wi_max, wi_num = -50, 0, 400

# threshold value below which a value is treated as zero
eps = 1e-3

if __name__=='__main__':
    print(globals())