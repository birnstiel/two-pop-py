#!/usr/bin/env python
"""
-------------------------------------------------------------------------------
         _____ _    _  _____       ______ ___________      ________   __
        |_   _| |  | ||  _  |      | ___ \  _  | ___ \     | ___ \ \ / /
          | | | |  | || | | |______| |_/ / | | | |_/ /_____| |_/ /\ V / 
          | | | |/\| || | | |______|  __/| | | |  __/______|  __/  \ /  
          | | \  /\  /\ \_/ /      | |   \ \_/ / |         | |     | |  
          \_/  \/  \/  \___/       \_|    \___/\_|         \_|     \_/  
                                                                              

This script runs a two-population dust model according to Birnstiel, Klahr,
Ercolano, A&A (2012). The output of the code is described in the README.md file.

Available at: https://github.com/birnstiel/two-pop-py

For bug reports, questions, ... contact birnstiel@mpia.de.

Note:
-----

If you use this code in a publication, please cite at least Birnstiel,
Klahr, & Ercolano, A&A (2012)[1], and possibly Birnstiel et al. (ApJL) 2015[2]
if you use the size distribution reconstruction. I addition to that, it would
be best practice to include the hash of the version you used to make sure
results are reproducible, as the code can change.

[1]: http://dx.doi.org/10.1051/0004-6361/201118136
[2]: http://dx.doi.org/10.1088/2041-8205/813/1/L14

------------------------------------------------------------------------------- 
"""
import gzip, bz2, os, sys
import pickle
from .args import args
from .results import results

compressors  = {
    'no match':[open,'raw'],'raw':[open,'raw'],
    'gzip':[gzip.GzipFile,'pgz'],'gz':[gzip.GzipFile,'pgz'],
    'bz2':[bz2.BZ2File,'pbz2']}
    
class task_status(object):
    """
    Context manager to show that a task is in progrees, or finished.
    
    Arguments:
    ----------
    
    start : string
        string description of the process
    
    Keywords:
    ---------
    
    end : string
        string for announcing completion of process, such as 'Done!'
        
    dots : string
        what to print in between, defaults to '...'
        
    blink : bool
        default: True; causes the dots to blink using ANSI control characters
    
    Example:
    --------
    
    >>> import time
    >>> with task_status('Loading') as t: time.sleep(5)
    Loading ... Done!
    
    """
    import sys
    def __init__(self, start, end='Done!', dots="...",blink=True):
        self.start = start
        self.end   = end
        self.dots  = dots
        self.blink = blink
    def __enter__(self):
        if self.blink:
            sys.stdout.write(self.start+' '+"\033[0;5m{}\033[0m".format(self.dots))
        else:
            sys.stdout.write(self.start+' '+self.dots)
        sys.stdout.flush()
    def __exit__(self, type, value, traceback):
        print('\r'+self.start+' '+self.dots+' '+self.end)


def get_compression_type(filename):
    """
    Pass a file name. It if looks like it's compressed with one of these
    
    - gz
    - bz2
    - zip
    
    that name will be returned.
    
    """
    magic_dict = {
        "\x1f\x8b\x08": "gz",
        "\x42\x5a\x68": "bz2",
        "\x50\x4b\x03\x04": "zip"
        }

    max_len = max(len(x) for x in magic_dict)
    
    try:
        with open(filename,'rb') as f: file_start = f.read(max_len)
            
        for magic, filetype in magic_dict.items():
            if file_start.startswith(magic):
                return filetype
        return "no match"
    except:
        # use the file ending
        d = dict(zip([l[1] for l in compressors.values()],compressors.keys()))
        ext = os.path.splitext(filename)[1][1:]
        return d[ext]

def load_grid_results(fname):
    """
    Load list of grid results from file
    """
    compressor,suffix = compressors[get_compression_type(fname)]
    
    with task_status('Loading {}-file \'{}\''.format(suffix,fname)), compressor(fname) as f: res=pickle.load(f)
    return res
    
def write_grid_results(res,fname,compression='gzip'):
    """
    Write list of grid results to file.
    
    Arguments:
    ----------
    
    res : list
        list of two_pop_run.results instances
        
    fname : string
        filename to write to
        
    Keywords:
    ---------
    
    compression : string
        possible compression mechanisms are 'raw', 'pgz', 'pbz2'.
    """
    if compression not in compressors.keys():
        raise NameError('{} is not a defined compression method'.format(compression))
    compressor,suffix = compressors[compression]
    fname = os.path.splitext(fname)[0]+os.path.extsep+suffix
    with task_status('Writing {}-file \'{}\''.format(suffix,fname)), compressor(fname,'w') as f: pickle.dump(res,f)

def lbp_solution(R,gamma,nu1,mstar,mdisk,RC0,time=0):
    """
    Calculate Lynden-Bell & Pringle self similar solution.
    All values need to be either given with astropy-units, or
    in as pure float arrays in cgs units.
    
    Arguments:
    ----------
    
    R : array
        radius array
    
    gamma : float
        viscosity exponent
    
    nu1 : float
        viscosity at R[0]
    
    mstar : float
        stellar mass
    
    mdisk : float
        disk mass at t=0
    
    RC0 : float
        critical radius at t=0
    
    Keywords:
    ---------
    
    time : float
        physical "age" of the analytical solution 
    
    Output:
    -------
    sig_g,RC(t)
    
    sig_g : array
        gas surface density, with or without unit, depending on input
    
    RC : the critical radius
    
    """
    import astropy.units as u
    import numpy as np

    # assume cgs if no units are given
    
    units = True
    if not hasattr(R,'unit'):
        R     = R*u.cm
        units  = False
    if not hasattr(mdisk,'unit'):
        mdisk = mdisk*u.g
    if not hasattr(mstar,'unit'):
        mstar = mstar*u.g
    if not hasattr(nu1,'unit'):
        nu1   = nu1*u.cm**2/u.s
    if not hasattr(RC0,'unit'):
        RC0   = RC0*u.cm
    if time is None: time = 0
    if not hasattr(time,'unit'):
        time  = time*u.s
        
    # convert to variables as in Hartmann paper
    
    R1   = R[0]
    r    = R/R1
    ts   = 1./(3*(2-gamma)**2)*R1**2/nu1
    
    T0   = (RC0/R1)**(2.-gamma)
    toff = (T0-1)*ts
    
    T1   = (time+toff)/ts+1
    RC1  = T1**(1./(2.-gamma))*R1
            
    # the normalization constant
    
    C  = (-3*mdisk*nu1*T0**(1./(4. - 2.*gamma))*(-2 + gamma))/2./R1**2
    
    # calculate the surface density
    
    sig_g = C/(3*np.pi*nu1*r)*T1**(-(5./2.-gamma)/(2.-gamma))*np.exp(-(r**(2.-gamma))/T1)
    
    if units:
        return sig_g,RC1
    else:
        return sig_g.cgs.value,RC1.cgs.value

def model_wrapper(ARGS,plot=False,save=False):
    """
    This is a wrapper for the two-population model `model.run`, in which
    the disk profile is a self-similar solution.
    
    Arguments:
    ----------
    ARGS : instance of the input parameter object
    
    Keywords:
    ---------
    
    plot : bool
          whether or not to plot the default figures

    save : bool
          whether or not to write the data to disk
    
    Output:
    -------
    results : instance of the results object
    """
    import numpy as np
    from . import model
    from matplotlib    import pyplot as plt
    from .const         import AU, year, Grav, k_b, mu, m_p
    from numbers       import Number
    #
    # set parameters according to input
    #
    nr       = ARGS.nr
    nt       = ARGS.nt
    tmax     = ARGS.tmax
    n_a      = ARGS.na
    alpha    = ARGS.alpha
    d2g      = ARGS.d2g
    mstar    = ARGS.mstar
    tstar    = ARGS.tstar
    rstar    = ARGS.rstar
    rc       = ARGS.rc
    rt       = ARGS.rt
    r0       = ARGS.r0
    r1       = ARGS.r1
    mdisk    = ARGS.mdisk
    rhos     = ARGS.rhos
    vfrag    = ARGS.vfrag
    a0       = ARGS.a0
    gamma    = ARGS.gamma
    edrift   = ARGS.edrift
    gasevol  = ARGS.gasevol
    tempevol = ARGS.tempevol
    starevol = ARGS.starevol
    T        = ARGS.T
    #
    # print setup
    #
    print(__doc__)
    print('\n'+48*'-')
    print(  'Model parameters:')
    ARGS.print_args()
    #
    # ===========
    # SETUP MODEL
    # ===========
    #
    # create grids and temperature
    #
    nri           = nr+1
    xi            = np.logspace(np.log10(r0),np.log10(r1),nri)
    x             = 0.5*(xi[1:]+xi[:-1])
    timesteps     = np.logspace(4,np.log10(tmax/year),nt)*year
    if starevol:
        raise ValueError('stellar evolution not implemented')

    # if T is not set, define default temperature function.
        
    if T is None:
        def T(x,locals_):
            return ( (0.05**0.25*tstar * (x /rstar)**-0.5)**4 + (7.)**4)**0.25
        
    # if temperature should not evolve, then replace the function with its initial value        
        
    if not tempevol and hasattr(T,'__call__'):
        T = T(x,locals())
    
    # set the initial surface density & velocity according Lynden-Bell & Pringle solution
    
    if isinstance(alpha, (list, tuple, np.ndarray)):
        alpha_fct = lambda x,locals_: alpha
        print('alpha given as array, ignoring gamma index when setting alpha')
    elif hasattr(alpha,'__call__'):
        alpha_fct = alpha
    elif isinstance(alpha,Number):
        alpha_fct = lambda x,locals_: alpha*(x/x[0])**(gamma-1)
    
    try:  
        # this one could break if alpha_function works only in model.run
        om1       = np.sqrt(Grav*args.mstar/x[0]**3)
        cs1       = np.sqrt(k_b*T[0]/mu/m_p)
        nu1       = alpha_fct(x,locals())*cs1**2/om1
        sigma_g,_ = lbp_solution(x, gamma, nu1, mstar, mdisk, rc)
        v_gas     = -3.0*alpha_fct(x,locals())*k_b*T/mu/m_p/2./np.sqrt(Grav*mstar/x)*(1.+7./4.)
    except:
        sigma_g   = mdisk*(2.-gamma)/(2.*np.pi*rc**2)*(x/rc)**-gamma*np.exp(-(x/rc)**(2.-gamma))
        v_gas     = np.zeros(sigma_g.shape)
        
    # truncation
        
    sigma_g[x>=rt] = 1e-100
        
    # normalize disk mass
     
    sigma_g = np.maximum(sigma_g,1e-100)
    sigma_g = sigma_g/np.trapz(2*np.pi*x*sigma_g,x=x)*mdisk
    sigma_d = sigma_g*d2g
    
    # call the model
    
    TI,SOLD,SOLG,VD,VG,v_0,v_1,a_dr,a_fr,a_df,a_t,Tout,alphaout = model.run(x,a0,timesteps,sigma_g,sigma_d,v_gas,T,alpha_fct,mstar,vfrag,rhos,edrift,nogrowth=False,gasevol=gasevol)
    #
    # ================================
    # RECONSTRUCTING SIZE DISTRIBUTION
    # ================================
    #
    a  = np.logspace(np.log10(a0),np.log10(5*a_t.max()),n_a)
    print('\n'+48*'-')
    if model.distri_available:
        try:
            print('reconstructing size distribution')
            reconstruct_size_distribution = model.reconstruct_size_distribution
            it = -1
            sig_sol,_,_,_,_,_ = reconstruct_size_distribution(x,a,TI[it],SOLG[it],SOLD[-1],alpha*np.ones(nr),rhos,T,mstar,vfrag,a_0=a0)
        except Exception:
            import traceback,warnings
            w = 'Could not reconstruct size distribution\nTraceback:\n----------\n' 
            w+= traceback.format_exc()  
            w+= '\n----------'
            warnings.warn(w)
            a       = None
            sig_sol = None
    else:
        print('distribution reconstruction is not available!')
    #
    # fill the results and write them out
    #    
    res = results()
    res.sigma_g   = SOLG
    res.sigma_d   = SOLD
    res.x         = x
    
    if hasattr(T,'__call__'):
        res.T = Tout
    else:
        res.T = T
        
    if hasattr(alpha,'__call__'):
        res.alpha = alphaout
    else:
        res.alpha = alpha
        
    res.timesteps = timesteps
    res.v_gas     = VG
    res.v_dust    = VD
    res.v_0       = v_0
    res.v_1       = v_1
    res.a_dr      = a_dr
    res.a_fr      = a_fr
    res.a_df      = a_df
    res.a_t       = a_t
    res.args      = ARGS
    res.a         = a
    
    if model.distri_available:
        res.sig_sol = sig_sol
        
    if save: res.write()
    #
    # ========
    # PLOTTING
    # ========
    #
    if plot:
        print(48*'-')
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
        
        if model.distri_available:
            _,ax = plt.subplots(tight_layout=True)
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
    
    print(48*'-'+'\n')
    print('ALL DONE'.center(48))
    print('\n'+48*'-')
    return res
    
def model_wrapper_test():
    """
    Test gas evolution: use small rc and large alpha
    """
    from .const import AU
    Args       = args()
    Args.rc    = 20*AU
    Args.alpha = 1e-2
    res        = model_wrapper(Args)
    return res

def model_wrapper_test_plot(res):
    """
    Plot the test result and returns the figure.
    """
    from .const import Grav, m_p, mu, k_b, year, AU
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import style
    style.use(['seaborn-dark',{'axes.grid': True,'font.size':10}]);
    
    # read the results
    
    args  = res.args
    x     = res.x
    sig_0 = res.sigma_g[0]
    sig_g = res.sigma_g[-1]
    t     = res.timesteps[-1]
    temp  = res.T
    alpha = args.alpha
    gamma = args.gamma
    rc    = args.rc
    mdisk = args.mdisk
    mstar = args.mstar
    
    # calculate analytical solution
    
    cs1 = np.sqrt(k_b*temp[0]/mu/m_p)
    om1 = np.sqrt(Grav*mstar/x[0]**3)
    nu1 = alpha*cs1**2/om1
    siga_0,_ = lbp_solution(x,gamma,nu1,mstar,mdisk,rc)
    siga_1,_ = lbp_solution(x,gamma,nu1,mstar,mdisk,rc,time=t)
    
    # compare results against analytical solution
    
    f,axs = plt.subplots(1,2,figsize=(10,4),sharex=True,sharey=True)
    axs[0].loglog(x/AU,siga_0,'-',label='analytical');
    axs[0].loglog(x/AU,sig_0,'r--',label='initial');
    axs[0].set_title('t = 0 years')
    axs[0].legend();
    axs[1].loglog(x/AU,siga_1,'-',label='analytical');
    axs[1].loglog(x/AU,sig_g,'r--',label='simulated');
    axs[1].set_title('t = {:3.2g} years'.format(t/year))
    axs[1].legend();
    axs[1].set_ylim(1e-5,1e5);
    for ax in axs:
        ax.set_xlabel('r [AU]')
        ax.set_ylabel('$\Sigma_\mathrm{g}$ [g cm$^{-2}$]');
    return f
    
def main():
    import argparse
    from . import const as c
    #
    # =================
    # ARGUMENT HANDLING
    # =================
    #
    # read in arguments
    #
    RTHF = argparse.RawTextHelpFormatter
    PARSER = argparse.ArgumentParser(description=__doc__,formatter_class=RTHF)
    PARSER.add_argument('-nr',    help='number of radial grid points',         type=int,   default=200)
    PARSER.add_argument('-nt',    help='number of snapshots',                  type=int  , default=100)
    PARSER.add_argument('-na',    help='number of particle sizes (use many!)', type=int  , default=150)
    PARSER.add_argument('-tmax',  help='simulation end time [yr]',             type=float, default=1e6)
    PARSER.add_argument('-alpha', help='turbulence parameter',                 type=float, default=1e-3)
    PARSER.add_argument('-d2g',   help='dust-to-gas ratio',                    type=float, default=1e-2)
    PARSER.add_argument('-mstar', help='stellar mass [solar masses]',          type=float, default=0.7)
    PARSER.add_argument('-tstar', help='stellar temperature [K]',              type=float, default=4010.)
    PARSER.add_argument('-rstar', help='stellar radius [solar radii]',         type=float, default=1.806)
    PARSER.add_argument('-rc',    help='disk characteristic radius [AU]',      type=float, default=200)
    PARSER.add_argument('-rt',    help='outer disk truncation radius [AU]',    type=float, default=1e6)
    PARSER.add_argument('-mdisk', help='disk mass in central star masses',     type=float, default=0.1)
    PARSER.add_argument('-rhos',  help='bulk density of the dusg [ g cm^-3]',  type=float, default=1.156)
    PARSER.add_argument('-vfrag', help='fragmentation velocity [ cm s^-1]',    type=float, default=1000)
    PARSER.add_argument('-a0',    help='initial grain size [cm]',              type=float, default=1e-5)
    PARSER.add_argument('-gamma', help='viscosity: alpha*(r/r[0])**gamma',     type=float, default=1)
    PARSER.add_argument('-edrift',help='drift fudge factor',                   type=float, default=1.0)
    PARSER.add_argument('-dir',   help='output directory default: data/',      type=str,   default='data')
    
    PARSER.add_argument('-p',               help='produce plots if possible',  action='store_true')
    PARSER.add_argument('-g','--gasevol',   help='turn *off* gas evolution',   action='store_false')
    ARGSIN = PARSER.parse_args()
    
    # convert units to cgs
    
    ARGSIN.tmax  *= c.year 
    ARGSIN.mstar *= c.M_sun
    ARGSIN.rstar *= c.R_sun
    ARGSIN.rc    *= c.AU
    ARGSIN.rt    *= c.AU
    ARGSIN.mdisk *= ARGSIN.mstar
    
    # convert to arguments object
    
    ARGS = args()
    for name,_ in ARGS.varlist:
        if hasattr(ARGSIN,name):
            setattr(ARGS,name,getattr(ARGSIN, name))
        
    # call the wrapper
    
    model_wrapper(ARGS,save=True,plot=ARGSIN.p)