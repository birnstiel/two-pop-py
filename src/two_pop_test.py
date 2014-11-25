#!/usr/bin/env python
from two_pop_model import two_pop_model_run
from constants import AU,year,Grav,M_sun,k_b,mu,m_p,R_sun
import os
from utilities import plotter
from matplotlib.pylab import * #@UnusedWildImport
#
# create the model
#
n_r       = 200
n_t       = 100
n_m       = 100
x         = logspace(log10(0.05),log10(4e3),n_r)*AU

alpha0        = 1e-3
M_star        = M_sun
T_star        = 4300.
R_star        = 2.5*R_sun
R_c           = 60*AU
M_disk        = 0.1*M_star
timesteps     = logspace(4,log10(3e6),n_t)*year
alpha         = alpha0*ones([n_t,n_r])
T             = ( (0.05**0.25*T_star * (x/R_star)**-0.5)**4 + 1e4)**0.25
T             = tile(T,[n_t,1])
peak_position = zeros(n_t)
m_star        = M_star*ones(n_t)
T_COAG_START  = 0.0
RHO_S         = 1.6
V_FRAG        = 1000
a_0           = 1e-4
E_drift       = 1.0
#
# set the initial surface density & velocity
#
profile     = maximum(M_disk/(2*pi*R_c**2)*(R_c/x)*exp(-x/R_c),1e-100)
sigma_g     = tile(profile,[n_t,1])
sigma_d     = tile(append(array(profile,ndmin=2)/100.,zeros([n_m-1,n_r]),0),[n_t,1])
v_gas       = -3.0*alpha*k_b*T/mu/m_p/2./sqrt(Grav*M_star/x)*(1.+7./4.)
#
# call the model
#
[TI,SOL,V,V_0,V_1,A_DR,A_FR,A_DF,A_T] = two_pop_model_run(x,a_0,timesteps,sigma_g,sigma_d,v_gas,T,alpha,m_star,
                              T_COAG_START,V_FRAG,RHO_S,peak_position,
                              E_drift)
#
# remove the line which shouldn't be there (I should fix this)
#         
SOL  = delete(SOL,1,0)
V    = delete(V,1,0)
V_0  = delete(V_0,1,0)
V_1  = delete(V_1,1,0)
A_DR = delete(A_DR,1,0)
A_FR = delete(A_FR,1,0)
A_DF = delete(A_DF,1,0) 
A_T  = delete(A_T,1,0) 
TI   = delete(TI,1,0) 
#
# show the evolution of the sizes
#
plotter(x=x/AU,data=A_FR,data2=A_DR,times=TI/year,xlog=1,ylog=1,xlim=[0.5,500],ylim=[2e-5,2e5],xlabel='r [AU]',i_start=0,ylabel='grain size [cm]')
#
# evolution of the surface density
#
plotter(x=x/AU,data=SOL,times=TI/year,xlog=1,ylog=1,xlim=[0.5,500],ylim=[2e-5,2e5],xlabel='r [AU]',i_start=0,ylabel='$\Sigma_d$ [g cm $^{-2}$]')	
#
# export the data
#
dirname = 'data'
if not os.path.isdir(dirname):
	os.mkdir(dirname)
savetxt(dirname+os.sep+'sigma_g.dat',sigma_g)
savetxt(dirname+os.sep+'sigma_d.dat',SOL)
savetxt(dirname+os.sep+'x.dat',x)
savetxt(dirname+os.sep+'T.dat',T)
savetxt(dirname+os.sep+'time.dat',timesteps)
savetxt(dirname+os.sep+'alpha.dat',alpha)
savetxt(dirname+os.sep+'v_gas.dat',v_gas)
savetxt(dirname+os.sep+'v_0.dat',V_0)
savetxt(dirname+os.sep+'v_1.dat',V_1)
savetxt(dirname+os.sep+'a_dr.dat',A_DR)
savetxt(dirname+os.sep+'a_fr.dat',A_FR)
savetxt(dirname+os.sep+'a_df.dat',A_DF)
savetxt(dirname+os.sep+'a_t.dat',A_T)

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

show()