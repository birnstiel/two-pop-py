"""
set some general constants in CGI units
"""
import numpy as np
import scipy.constants as sc

pi           = np.pi;                   # PI
k_b          = sc.k*1e7;                # Boltzmann constant in erg/K
m_p          = sc.proton_mass*1e3;      # proton mass in g
Grav         = sc.G*1e3;                # gravitational constant in cm^3 g^-1 s^-2
AU           = sc.au*1e2;               # astronomical unit in cm
year         = sc.Julian_year;          # year in s
mu           = 2.3e0;                   # mean molecular mass in proton masses
M_sun        = 1.9891e+33;              # mass of the sun in g
R_sun        = 69550800000.0;           # radius of the sun in cm