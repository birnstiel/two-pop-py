#!/usr/bin/env python
"""
         _____ _    _  _____       ______ ___________      ________   __
        |_   _| |  | ||  _  |      | ___ \  _  | ___ \     | ___ \ \ / /
          | | | |  | || | | |______| |_/ / | | | |_/ /_____| |_/ /\ V / 
          | | | |/\| || | | |______|  __/| | | |  __/______|  __/  \ /  
          | | \  /\  /\ \_/ /      | |   \ \_/ / |         | |     | |  
          \_/  \/  \/  \___/       \_|    \___/\_|         \_|     \_/  
                                                                              

This script runs a two-population dust model according to Birnstiel, Klahr,
Ercolano, A&A (2012).

Available at: https://github.com/birnstiel/two-pop-py

For bug reports, questions, ... contact birnstiel@mpia.de. 
"""
import os, argparse
import numpy as np
import two_pop_model
from matplotlib    import pyplot as plt
from constants     import AU, year, Grav, M_sun, k_b, mu, m_p, R_sun, pi
from two_pop_model import two_pop_model_run
#
# =================
# ARGUMENT HANDLING
# =================
#
# read in arguments
#
RTHF = argparse.RawTextHelpFormatter
PARSER = argparse.ArgumentParser(description=__doc__,formatter_class=RTHF)
PARSER.add_argument('-nr',    help=' number of radial grid points',         type=int,   default=200)
PARSER.add_argument('-nt',    help=' number of snapshots',                  type=int  , default=100)
PARSER.add_argument('-na',    help=' number of particle sizes (use many!)', type=int  , default=150)
PARSER.add_argument('-tmax',  help=' simulation end time [yr]',             type=float, default=3e6)
PARSER.add_argument('-alpha', help=' turbulence parameter',                 type=float, default=1e-3)
PARSER.add_argument('-d2g',   help=' dust-to-gas ratio',                    type=float, default=1e-2)
PARSER.add_argument('-mstar', help=' stellar mass [solar masses]',          type=float, default=1.0)
PARSER.add_argument('-tstar', help=' stellar temperature [K]',              type=float, default=4300.)
PARSER.add_argument('-rstar', help=' stellar radius [solar radii]',         type=float, default=2.5)
PARSER.add_argument('-rc',    help=' disk characteristic radius [AU]',      type=float, default=60)
PARSER.add_argument('-mdisk', help=' disk mass in central star masses',     type=float, default=0.1)
PARSER.add_argument('-rhos',  help=' bulk density of the dusg [ g cm^-3]',  type=float, default=1.6)
PARSER.add_argument('-vf',    help=' fragmentation velocity [ cm s^-1]',    type=float, default=1000)
PARSER.add_argument('-a0',    help=' initial grain size [cm]',              type=float, default=1e-4)
PARSER.add_argument('-edrift',help=' drift fudge factor',                   type=float, default=1.0)
PARSER.add_argument('-dir',   help=' output directory default: data/',      type=str,   default='data')
PARSER.add_argument('-p',     help=' produce plots if possible',            action='store_true')
ARGS = PARSER.parse_args()
#
# set parameters according to input
#
n_r     = ARGS.nr
n_t     = ARGS.nt
t_max   = ARGS.tmax*year
n_a     = ARGS.na
alpha   = ARGS.alpha
d2g     = ARGS.d2g
M_star  = ARGS.mstar*M_sun
T_star  = ARGS.tstar
R_star  = ARGS.rstar*R_sun
R_c     = ARGS.rc*AU
M_disk  = ARGS.mdisk*M_star
RHO_S   = ARGS.rhos
V_FRAG  = ARGS.vf
a_0     = ARGS.a0
E_drift = ARGS.edrift
dirname = ARGS.dir
plot    = ARGS.p
#
# print setup
#
print(__doc__)
print('\n'+35*'-')
print(  'Model parameters:')
print(  35*'-'+'\n')
printvals = [
            ['n_r',n_r,''],
            ['n_t',n_t,''],
            ['t_max',t_max/year,'years'],
            ['alpha',alpha,''],
            ['d2g',d2g,''],
            ['M_star',M_star/M_sun,'solar masses'],
            ['T_star',T_star,'K'],
            ['R_star',R_star/R_sun,'R_sun'],
            ['R_c',R_c/AU,'AU'],
            ['M_disk',M_disk/M_star,'M_star'],
            ['RHO_S',RHO_S,'g/cm^3'],
            ['V_FRAG',V_FRAG,'cm/s'],
            ['a_0',a_0,'cm'],
            ['E_drift',E_drift,'']
            ]

for n,v,u in printvals:
    print(n.ljust(8)+' = '+'{:3.2g}'.format(v).rjust(10)+' '+u)
print('\n'+35*'-')
#
# ===========
# SETUP MODEL
# ===========
#
# create grids and temperature
#
n_ri          = n_r+1
xi            = np.logspace(np.log10(0.05),np.log10(4e3),n_ri)*AU
x             = 0.5*(xi[1:]+xi[:-1])
timesteps     = np.logspace(4,np.log10(t_max/year),n_t)*year
T             = ( (0.05**0.25*T_star * (x /R_star)**-0.5)**4 + 1e4)**0.25
#
# set the initial surface density & velocity
#
sigma_g     = np.maximum(M_disk/(2*pi*R_c**2)*(R_c/x)*np.exp(-x/R_c),1e-100)
sigma_d     = sigma_g*d2g
v_gas       = -3.0*alpha*k_b*T/mu/m_p/2./np.sqrt(Grav*M_star/x)*(1.+7./4.)
#
# call the model
#
[TI,SOLD,SOLG,VD,VG,v_0,v_1,a_dr,a_fr,a_df,a_t] = two_pop_model_run(x,a_0,timesteps,sigma_g,sigma_d,v_gas,T,alpha*np.ones(n_r),M_star,V_FRAG,RHO_S,E_drift,nogrowth=False)
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
# ================================
# RECONSTRUCTING SIZE DISTRIBUTION
# ================================
#
print('\n'+35*'-')
if two_pop_model.distri_available:
    print('reconstructing size distribution')
    reconstruct_size_distribution = two_pop_model.reconstruct_size_distribution
    it = -1
    a  = np.logspace(np.log10(a_0),np.log10(5*a_t.max()),n_a)
    sig_sol,a_max,r_f,_,_,_ = reconstruct_size_distribution(x,a,TI[it],SOLG[it],SOLD[-1],alpha*np.ones(n_r),RHO_S,T,M_star,V_FRAG,a_0=a_0)
else:
    print('distribution reconstruction is not available!')
#
# ===========
# EXPORT DATA
# ===========
#
print('\n'+35*'-')
print('writing results to data/ ...')
if not os.path.isdir(dirname):
    os.mkdir(dirname)
np.savetxt(dirname+os.sep+'sigma_g.dat',SOLG)
np.savetxt(dirname+os.sep+'sigma_d.dat',SOLD)
np.savetxt(dirname+os.sep+'x.dat',x)
np.savetxt(dirname+os.sep+'T.dat',T)
np.savetxt(dirname+os.sep+'time.dat',timesteps)
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
dummy('alpha'.ljust(strl) +('=\t%g'%(alpha )).ljust(strl)+'# alpha [-]\n')
fid.close()

if two_pop_model.distri_available:
    np.savetxt(dirname+os.sep+'a.dat',a)
    np.savetxt(dirname+os.sep+'sigma_d_a.dat',sig_sol)
#
# ========
# PLOTTING
# ========
#
if plot:
    print(35*'-')
    print('plotting results ...') 
    try:
        from widget import plotter
        #
        # show the evolution of the sizes
        #
        plotter(x=x/AU,data=a_fr,data2=a_dr,times=TI/year,xlog=1,ylog=1,xlim=[0.5,500],ylim=[2e-5,2e5],xlabel='r [AU]',i_start=0,ylabel='grain size [cm]')
        #
        # evolution of the surface density
        #
        plotter(x=x/AU,data=SOLD,data2=SOLG,times=TI/year,xlog=1,ylog=1,xlim=[0.5,500],ylim=[2e-5,2e5],xlabel='r [AU]',i_start=0,ylabel='$\Sigma_d$ [g cm $^{-2}$]')
    except ImportError:
        print('Could not import GUI, will not plot GUI')
    
    if two_pop_model.distri_available:
        f,ax = plt.subplots(tight_layout=True)
        gsf  = 2*(a[1]/a[0]-1)/(a[1]/a[0]+1)
        mx   = np.ceil(np.log10(sig_sol.max()/gsf))
        cc=ax.contourf(x/AU,a,np.log10(np.maximum(sig_sol/gsf,1e-100)),np.linspace(mx-10,mx,50),cmap='OrRd')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('radius [AU]')
        ax.set_ylabel('particle size [cm]')
        cb = plt.colorbar(cc)
        cb.set_ticks(np.arange(mx-10,mx+1))
        cb.set_label('$a\cdot\Sigma_\mathrm{d}(r,a)$ [g cm$^{-2}$]')
    plt.show()

print(35*'-'+'\n')
print('ALL DONE'.center(35))
print('\n'+35*'-')