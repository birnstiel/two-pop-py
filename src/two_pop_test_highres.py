#/usr/bin/env python
from pydisk1D import pydisk1D
from numpy import * #@UnusedWildImport
from matplotlib.pylab import * #@UnusedWildImport
from two_pop_model import * #@UnusedWildImport
from get_sims import get_sims
from constants import * #@UnusedWildImport
import os,csv
from two_pop_model import * #@UnusedWildImport
#
# read in the first one
#
s=os.path.expanduser(get_sims()[0])
execfile('/Users/til/Dropbox/python-projects/pydisk1D/src/pydisk1D_interactive.py')
d=pydisk1D(s)
x = d.x
grainsizes = d.grainsizes
n_m = d.n_m
n_r = d.n_r
n_t = d.n_t
timesteps = d.timesteps
sigma_g = d.sigma_g
sigma_d = d.sigma_d
alpha = d.alpha
v_gas = d.v_gas
T = d.T
peak_position = d.peak_position
m_star = d.m_star
T_COAG_START = d.nml['T_COAG_START']
RHO_S = d.nml['RHO_S']
V_FRAG = d.nml['V_FRAG']
DRIFT_FUDGE_FACTOR = d.nml['DRIFT_FUDGE_FACTOR']
#
# calculate a_max for each (r,t)
#
test = 0
a_0 = grainsizes[find(sigma_d[0:n_m,round(n_r/2.)]>1e-50)[-1]]
sigma_d_total = d.get_sigma_dust_total()
#
# make a finer grid
#
n_r2 = 100
x_2             = logspace(log10(x[0]),log10(x[-1]),n_r2)
sigma_d_2       = zeros([n_m*n_t,n_r2])
sigma_g_2       = zeros([n_t,n_r2])
v_gas_2         = zeros([n_t,n_r2])
sigma_d_total_2 = zeros([n_t,n_r2])
T_2             = zeros([n_t,n_r2])
alpha_2         = zeros([n_t,n_r2])
peak_position_2 = zeros(n_t)
for it in arange(n_t):
    sigma_g_2[it,:]        = 10**interp(log10(x_2),log10(x),log10(sigma_g[it,:]))
    v_gas_2[it,:]          = interp(x_2,x,v_gas[it,:])
    sigma_d_total_2[it,:]  = 10**interp(log10(x_2),log10(x),log10(sigma_d_total[it,:]))
    sigma_d_2[it*n_m,:]    = sigma_d_total_2[it,:]
    T_2[it,:]              = 10**interp(log10(x_2),log10(x),log10(T[it,:]))
    alpha_2[it,:]          = 10**interp(log10(x_2),log10(x),log10(alpha[it,:]))
    peak_position_2[it]    = find(x_2>=x[peak_position[it]])[0]
#
# run the model
#
[T_h,SOL_h,V_h] = two_pop_model_run(x_2,a_0,timesteps,sigma_g_2,sigma_d_2,v_gas_2,T_2,alpha_2,m_star,T_COAG_START,V_FRAG,RHO_S,peak_position_2,DRIFT_FUDGE_FACTOR,plotting=True)
[T,SOL,V] = two_pop_model_run(x,a_0,timesteps,sigma_g,sigma_d,v_gas,T,alpha,m_star,T_COAG_START,V_FRAG,RHO_S,peak_position,DRIFT_FUDGE_FACTOR,plotting=True)
#
# load comaparison data
#
ds3=array([[float(i),float(j)] for i,j in csv.reader(open('./src/comparison_data/d2g_sim_1e-3_3e6.csv'))])
dt3=array([[float(i),float(j)] for i,j in csv.reader(open('./src/comparison_data/d2g_toy_1e-3_3e6.csv'))])
#
# plot comaparison
#
figure()
loglog(d.x/AU,d.get_sigma_dust_total()[-1]/d.sigma_g[-1],'k',label='from simulation')
#loglog(ds3[:,0],ds3[:,1],label='simulation digitized')
loglog(dt3[:,0],dt3[:,1],'r',label='toy digitized')
loglog(x_2/AU,SOL[-1]/sigma_g_2[-1],'g-', label='new toy')
loglog(x_2/AU,SOL_h[-1]/sigma_g_2[-1],'g--',label='new toy, highres')
ylim(2e-5,2e-1)
xlim(.5,500)
legend()