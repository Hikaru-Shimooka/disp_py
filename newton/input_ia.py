# parameter settings

# mass
mi = 1  # ion
mr = 1/1836  # mass ratio
me = mi*mr  # electron

# temperature
Ti = 1  # ion
# Te = Ti  # electron
# Te = 10*Ti  # electron
Te = 100*Ti  # electron

# plasma frequency
wpi = 1  # ion
wpe = wpi*((mi/me)**(1/2))  # electron

# debye wave number
ki = wpi*((mi/Ti)**(1/2))  # ion
ke = ki*((Ti/Te)**(1/2))  # electron

# thermal velocity
ve = (Te/me)**(1/2)  # electron
vi = (Ti/mi)**(1/2)  # ion

# k range
k_min, k_max, k_num = 0, 1, 200

# wr range
wr_min, wr_max, wr_num = 0, 2, 500

# wi range
wi_min, wi_max, wi_num = -1, 0, 500

# threshold value below which a value is treated as zero
# eps = 1e-2  # Te = Ti, Te = 10*Ti
eps = 1e-3  # Te = 100*Ti
