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
arcsec       = (pi/180./3600.)		    # 1 arcsecond in radian
arcsec_sq    = (pi/180./3600.)**2	    # 1 square arcsecond in sterad
sig_sb       = sc.Stefan_Boltzmann*1e3; # Stefan-Boltzmann constant in g s^-3 K^-4
h_planck     = sc.h*1e7;                # Planck's constant in erg s (   =g cm^2 / s)
c_light      = sc.c*1e2;                # speed of light in cm/s
year         = sc.Julian_year;          # year in s
PC           = sc.parsec*1e2;           # parsec in cm
ee           = sc.e*c_light/10.;        # elementary charge in Fr
eV           = sc.eV*1e7                # 1 electron volt in erg
P_triple     = 6116.57;                 # water triple point pressure in barye    = g/(s*cm)    = 0.1 Pa
T_triple     = 273.16;                  # water triple point Temperature in K
sig_h2       = 2e-15;                   # cross section of H2 [cm^2]
sig_h_atomar = 6e-18;                   # cross section of H
mu           = 2.3e0;                   # mean molecular mass in proton masses
M_sun        = 1.9891e+33;              # mass of the sun in g
M_mercury    = 3.302e26;                # Mercury mass in g
M_venus      = 4.869e27;                # venus   mass in g
M_earth      = 5.9742e+27;              # Earth   mass in g
M_mars       = 6.419e+26;               # Mars    mass in g
M_jupiter    = 1.8987e+30;              # Jupiter mass in g
M_saturn     = 5.685e29;                # Saturn  mass in g
M_uranus     = 8.683e28;                # Uranus  mass in g
M_neptune    = 1.0243e29;               # Neptune mass in g
M_pluto      = 1.25e25;                 # Pluto mass in g
M_moon       = 7.3459e25;               # Moon    mass in g
R_sun        = 69550800000.0;           # radius of the sun in cm
T_sun        = 5780.0;                  # Effective temperature of the sun in K
R_earth      = 637813600.0;             # equatorial radius of earth in cm
R_jupiter    = 7149200000.0;            # equatorial radius of jupiter in cm
L_sun        = 3.846e+33;               # solar luminosity in erg/s