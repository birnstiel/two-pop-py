#!/usr/bin/env python
"""
This script runs a two-population dust model according to Birnstiel, Klahr, Ercolano, ApJ (2012).
"""
import os
import numpy as np
from matplotlib    import pyplot as plt
from constants     import AU,year,Grav,M_sun,k_b,mu,m_p,R_sun,pi
from two_pop_model import two_pop_model_run
from widget        import plotter
#
# ===========
# SETUP MODEL
# ===========
#
# set parameters
#
n_r           = 200         # number of radial grid points
n_t           = 100         # number of snapshots
alpha         = 1e-3        # turbulence parameter
d2g           = 1e-2        # dust-to-gas ratio
M_star        = M_sun       # stellar mass [g]
T_star        = 4300.       # stellar temperature [K]
R_star        = 2.5*R_sun   # stellar radius [K]
R_c           = 60*AU       # disk characteristic radius [cm]
M_disk        = 0.1*M_star  # disk mass [g]
RHO_S         = 1.6         # bulk density of the dusg [ g cm^-3]
V_FRAG        = 1000        # fragmentation velocity [ cm s^-1]
a_0           = 1e-4        # initial grain size [cm]
E_drift       = 1.0         # drift fudge factor
#
# create grids and temperature
#
x             = np.logspace(np.log10(0.05),np.log10(4e3),n_r)*AU
timesteps     = np.logspace(4,np.log10(3e6),n_t)*year
T             = ( (0.05**0.25*T_star * (x/R_star)**-0.5)**4 + 1e4)**0.25
peak_position = np.zeros(n_t)
#
# set the initial surface density & velocity
#
sigma_g     = np.maximum(M_disk/(2*pi*R_c**2)*(R_c/x)*np.exp(-x/R_c),1e-100)
sigma_d     = sigma_g*d2g
v_gas       = -3.0*alpha*k_b*T/mu/m_p/2./np.sqrt(Grav*M_star/x)*(1.+7./4.)
#
# call the model
#
[TI,SOLD,SOLG,VD,VG,v_0,v_1,a_dr,a_fr,a_df,a_t] = two_pop_model_run(x,a_0,timesteps,sigma_g,sigma_d,v_gas,T,alpha*np.ones(n_r),M_star,V_FRAG,RHO_S,peak_position,E_drift,nogrowth=False)
#
# remove the line which shouldn't be there (I should fix this)
#         
TI   = np.delete(TI,1,0) 
SOLD = np.delete(SOLD,1,0)
SOLG = np.delete(SOLG,1,0)
VD   = np.delete(VD,1,0)
VG   = np.delete(VG,1,0)
v_0  = np.delete(v_0,1,0)
v_1  = np.delete(v_1,1,0)
a_dr = np.delete(a_dr,1,0)
a_fr = np.delete(a_fr,1,0)
a_df = np.delete(a_df,1,0) 
a_t  = np.delete(a_t,1,0) 
#
# show the evolution of the sizes
#
plotter(x=x/AU,data=a_fr,data2=a_dr,times=TI/year,xlog=1,ylog=1,xlim=[0.5,500],ylim=[2e-5,2e5],xlabel='r [AU]',i_start=0,ylabel='grain size [cm]')
#
# evolution of the surface density
#
plotter(x=x/AU,data=SOLD,data2=SOLG,times=TI/year,xlog=1,ylog=1,xlim=[0.5,500],ylim=[2e-5,2e5],xlabel='r [AU]',i_start=0,ylabel='$\Sigma_d$ [g cm $^{-2}$]')	
#
# export the data
#
dirname = 'data'
if not os.path.isdir(dirname):
	os.mkdir(dirname)
np.savetxt(dirname+os.sep+'sigma_g.dat',SOLG)
np.savetxt(dirname+os.sep+'sigma_d.dat',SOLD)
np.savetxt(dirname+os.sep+'x.dat',x)
np.savetxt(dirname+os.sep+'T.dat',T)
np.savetxt(dirname+os.sep+'time.dat',timesteps)
np.savetxt(dirname+os.sep+'alpha.dat',alpha)
np.savetxt(dirname+os.sep+'v_gas.dat',v_gas)
np.savetxt(dirname+os.sep+'v_0.dat',v_0)
np.savetxt(dirname+os.sep+'v_1.dat',v_1)
np.savetxt(dirname+os.sep+'a_dr.dat',a_dr)
np.savetxt(dirname+os.sep+'a_fr.dat',a_fr)
np.savetxt(dirname+os.sep+'a_df.dat',a_df)
np.savetxt(dirname+os.sep+'a_t.dat',a_t)

fid = open(dirname+os.sep+'constants.dat','w')
strl = 15
dummy = fid.write
dummy('n_r'.ljust(strl)   +('=\t%i'%(n_r   )).ljust(strl)+'# number of radial grid points\n')
dummy('n_t'.ljust(strl)   +('=\t%i'%(n_t   )).ljust(strl)+'# number of snapshots\n')
dummy('M_star'.ljust(strl)+('=\t%g'%(M_star)).ljust(strl)+'# stellar mass [M_sun]\n')
dummy('RHO_S'.ljust(strl) +('=\t%g'%(RHO_S )).ljust(strl)+'# dust internal density [g cm^-3]\n')
dummy('a_0'.ljust(strl)   +('=\t%g'%(a_0   )).ljust(strl)+'# size of smallest grains [cm]\n')
dummy('mu'.ljust(strl)    +('=\t%g'%(mu    )).ljust(strl)+'# mean molecular weight [proton masses]\n')
fid.close()

plt.show()