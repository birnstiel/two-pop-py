"""
set some general constants in CGI units
"""
import numpy as np
import scipy.constants as sc

pi           = np.pi;                   # noqa - PI
k_b          = sc.k*1e7;                # noqa - Boltzmann constant in erg/K
m_p          = sc.proton_mass*1e3;      # noqa - proton mass in g
Grav         = sc.G*1e3;                # noqa - gravitational constant in cm^3 g^-1 s^-2
AU           = sc.au*1e2;               # noqa - astronomical unit in cm
year         = sc.Julian_year;          # noqa - year in s
mu           = 2.3e0;                   # noqa - mean molecular mass in proton masses
M_sun        = 1.9891e+33;              # noqa - mass of the sun in g
R_sun        = 69550800000.0;           # noqa - radius of the sun in cm
sig_h2       = 2e-15;                   # noqa - cross section of H2 [cm^2]
