#!/usr/bin/env python
from two_pop_model import two_pop_model_run
from constants import AU,year,Grav,M_sun,k_b,mu,m_p
import widget,os,csv
from pydisk1D import pydisk1D
from matplotlib.pylab import * #@UnusedWildImport
from get_sims import get_sims
#from numpy import logspace,log10,ones,tile,sqrt,zeros,maximum,delete,savetxt
#
# models can be:
# - power-law
# - read-in
#
#model = 'read-in'
model = 'power-law'
#
# a simple power-law model
#
if model == 'power-law':
	#
	# create the model
	#
	n_r       = 200
	n_t       = 100
	n_m       = 100
	x         = logspace(-1,log10(5e2),n_r)*AU
	
	alpha0        = 1e-3
	M_star        = M_sun
	timesteps     = logspace(4,log10(3e6),n_t)*year
	alpha         = alpha0*ones([n_t,n_r])
	T             = tile(100*(x/AU)**-0.5,[n_t,1])
	peak_position = zeros(n_t)
	m_star        = M_star*ones(n_t)
	T_COAG_START  = 0.0
	RHO_S         = 1.6
	V_FRAG        = 1000
	a_0           = 1e-4
	E_drift       = 1.0
	#
	# set the surface density & velocity
	#
	profile     = maximum(200*(x/AU)**-1,1e-100)
	sigma_g     = tile(profile,[n_t,1])
	sigma_d     = tile(append(array(profile,ndmin=2)/100.,zeros([n_m-1,n_r]),0),[n_t,1])
	v_gas       = -3.0*alpha*k_b*T/mu/m_p/2./sqrt(Grav*M_star/x)*(1.+7./4.)
#
# read in form the given diskev simulation
#
elif model == 'read-in':
	#
	# read in the simulation
	#
	s=os.path.expanduser(get_sims()[0])
	execfile('/Users/til/Dropbox/python-projects/pydisk1D/src/pydisk1D_interactive.py')
	d=pydisk1D(s)
	#
	# define the arrays
	#
	x             = d.x
	n_m           = d.n_m
	n_r           = d.n_r
	n_t           = d.n_t
	timesteps     = d.timesteps
	alpha         = d.alpha
	T             = d.T
	grainsizes    = d.grainsizes
	peak_position = d.peak_position
	m_star        = d.m_star
	M_star        = m_star[0]
	T_COAG_START  = d.nml['T_COAG_START']
	RHO_S         = d.nml['RHO_S']
	V_FRAG        = d.nml['V_FRAG']
	E_drift       = 1.0
	#
	# set the surface density profiles & velocity
	#
	sigma_g   = d.sigma_g.copy()
	sigma_d   = d.sigma_d.copy()
	sigma_d_t = d.get_sigma_dust_total()
	v_gas     = d.v_gas
	a_0 = grainsizes[find(sigma_d[0:n_m,round(n_r/2.)]>1e-50)[-1]]
else:
	print('ERROR: unknown model')
	sys.exit(1)
	
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
# show comparison to original code
#
if model=='read-in':
	#
	# read in the comaparison data
	#
	ds3=array([[float(i),float(j)] for i,j in csv.reader(open('./src/comparison_data/d2g_sim_1e-3_3e6.csv'))])
	dt3=array([[float(i),float(j)] for i,j in csv.reader(open('./src/comparison_data/d2g_toy_1e-3_3e6.csv'))])
	#
	# plot it
	#
	figure()
	loglog(d.x/AU,d.get_sigma_dust_total()[-1]/d.sigma_g[-1],label='from simulation')
	loglog(ds3[:,0],ds3[:,1],label='simulation digitized')
	loglog(dt3[:,0],dt3[:,1],label='toy digitized')
	loglog(x/AU,SOL[-1]/sigma_g[-1],label='new toy')
	ylim(2e-5,2e-1)
	xlim(.5,500)
	legend()
	#
	# show the evolution of the sizes
	#
	it1 = find(timesteps>=T_COAG_START)[0]
	widget.plotter(x=x/AU,data=SOL,data2=sigma_d_t[it1:,:],times=timesteps[it1:]/year,xlog=1,ylog=1,xlim=[0.5,500],ylim=[2e-5,2e5],xlabel='r [AU]',i_start=0,ylabel='$\Sigma_d$ [g cm $^{-2}$]')
	widget.plotter(x=x/AU,data=A_FR,data2=A_DR,times=TI/year,xlog=1,ylog=1,xlim=[0.5,500],ylim=[2e-5,2e5],xlabel='r [AU]',i_start=0,ylabel='$\Sigma_d$ [g cm $^{-2}$]')
if model =='power-law':
	#
	# show the evolution of the sizes
	#
	widget.plotter(x=x/AU,data=A_FR,data2=A_DR,times=TI/year,xlog=1,ylog=1,xlim=[0.5,500],ylim=[2e-5,2e5],xlabel='r [AU]',i_start=0,ylabel='$\Sigma_d$ [g cm $^{-2}$]')
	#
	# evolution of the surface density
	#
	#widget.plotter(x=x/AU,data=SOL,times=TI/year,xlog=1,ylog=1,xlim=[0.5,500],ylim=[2e-5,2e5],xlabel='r [AU]',i_start=0,ylabel='$\Sigma_d$ [g cm $^{-2}$]')	
#
#
#
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