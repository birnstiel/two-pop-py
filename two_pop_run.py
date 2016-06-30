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

class results:
    nri          = None
    xi           = None
    x            = None
    timesteps    = None
    T            = None
    sigma_g      = None
    sigma_d      = None
    v_gas        = None
    v_dust       = None
    v_0          = None
    v_1          = None
    a_dr         = None
    a_fr         = None
    a_df         = None
    a_t          = None
    args         = None
    a            = None
    sig_sol      = None
    
    def write(self,dirname=None):
        """
        Export data to the specified folder.
        """
        import os
        import numpy as np
        import two_pop_model
        
        if dirname is None: dirname = args.dir
        
        print('\n'+35*'-')
        print('writing results to {} ...'.format(dirname))
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        np.savetxt(dirname+os.sep+'sigma_g.dat', self.sigma_g)
        np.savetxt(dirname+os.sep+'sigma_d.dat', self.sigma_d)
        np.savetxt(dirname+os.sep+'x.dat',       self.x)
        np.savetxt(dirname+os.sep+'T.dat',       self.T)
        np.savetxt(dirname+os.sep+'time.dat',    self.timesteps)
        np.savetxt(dirname+os.sep+'v_gas.dat',   self.v_gas)
        np.savetxt(dirname+os.sep+'v_dust.dat',  self.v_dust)
        np.savetxt(dirname+os.sep+'v_0.dat',     self.v_0)
        np.savetxt(dirname+os.sep+'v_1.dat',     self.v_1)
        np.savetxt(dirname+os.sep+'a_dr.dat',    self.a_dr)
        np.savetxt(dirname+os.sep+'a_fr.dat',    self.a_fr)
        np.savetxt(dirname+os.sep+'a_df.dat',    self.a_df)
        np.savetxt(dirname+os.sep+'a_t.dat',     self.a_t)
        
        if two_pop_model.distri_available:
            np.savetxt(dirname+os.sep+'a.dat',         self.a)
            np.savetxt(dirname+os.sep+'sigma_d_a.dat', self.sig_sol)
        
        self.args.write_args()

    def read(self,dirname=None):
        """
        Read results from the specified folder.
        """
        import os
        import numpy as np
        
        if dirname is None:
            dirname = self.abs.dir
            
        
        print('\n'+35*'-')
        print('writing results to {} ...'.format(dirname))
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        self.sigma_g   = np.loadtxt(dirname+os.sep+'sigma_g.dat')
        self.sigma_d   = np.loadtxt(dirname+os.sep+'sigma_d.dat')
        self.x         = np.loadtxt(dirname+os.sep+'x.dat')
        self.T         = np.loadtxt(dirname+os.sep+'T.dat')
        self.timesteps = np.loadtxt(dirname+os.sep+'time.dat')
        self.v_gas     = np.loadtxt(dirname+os.sep+'v_gas.dat')
        self.v_0       = np.loadtxt(dirname+os.sep+'v_0.dat')
        self.v_1       = np.loadtxt(dirname+os.sep+'v_1.dat')
        self.a_dr      = np.loadtxt(dirname+os.sep+'a_dr.dat')
        self.a_fr      = np.loadtxt(dirname+os.sep+'a_fr.dat')
        self.a_df      = np.loadtxt(dirname+os.sep+'a_df.dat')
        self.a_t       = np.loadtxt(dirname+os.sep+'a_t.dat')
        
        if os.path.isfile(dirname+os.sep+'a.dat'):
            self.a       = np.savetxt(dirname+os.sep+'a.dat')
        if os.path.isfile(dirname+os.sep+'sigma_d_a.dat'):
            self.sig_sol = np.savetxt(dirname+os.sep+'sigma_d_a.dat')
        
        self.args = args()
        args.read(dirname=dirname)
    
class args:
    import const as _c
    
    # names of all parameters

    varlist = [ ['nr',     int],
    			['nt',     int],
    			['tmax',   float],
    			['alpha',  float],
    			['d2g',    float],
    			['mstar',  float],
    			['tstar',  float],
    			['rstar',  float],
    			['rc',     float],
    			['mdisk',  float],
    			['rhos',   float],
    			['vfrag',  float],
    			['a0',     float],
    			['gamma',  float],
    			['edrift', float]]
    
    # set default values
    
    nr      = 200
    nt      = 100
    na      = 150
    tmax    = 1e6*_c.year
    alpha   = 1e-3
    d2g     = 1e-2
    mstar   = 0.7*_c.M_sun
    tstar   = 4010.
    rstar   = 1.806*_c.R_sun
    rc      = 200*_c.AU
    mdisk   = 0.1*mstar
    rhos    = 1.156
    vfrag   = 1000
    a0      = 1e-5
    gamma   = 1.0
    edrift  = 1.0
    dir     = 'data'
    gasevol = True
    
    def __init__(self,**kwargs):
        """
        Initialize arguments. Simulation parameters can be given as keywords.
        """
        import warnings
        for k,v in kwargs.iteritems():
            if hasattr(self,k):
                setattr(self,k,v)
            else:
                warnings.warn("No such argument")
    
    def print_args(self):
        """
        Prints out all arguments
        """
        from const import year, M_sun, R_sun, AU
        print(  35*'-'+'\n')
        
        conversion = {
            'nr':      [1,            ''],
            'nt':      [1,            ''],
            'tmax':    [1/year,       'years'],
            'alpha':   [1,            ''],
            'd2g':     [1,            ''],
            'mstar':   [1/M_sun,      'solar masses'],
            'tstar':   [1,            'K'],
            'rstar':   [1/R_sun,      'R_sun'],
            'rc':      [1/AU,         'AU'],
            'mdisk':   [1/self.mstar, 'M_star'],
            'rhos':    [1,            'g/cm^3'],
            'vfrag':   [1,            'cm/s'],
            'a0':      [1,            'cm'],
            'gamma':   [1,            ''],
            'edrift':  [1,            '']
            }
        
        for n,conv_unit in conversion.iteritems():
            conv,unit = conv_unit
            print(n.ljust(9)+' = '+'{:3.2g}'.format(conv*getattr(self, n)).rjust(10)+' '+unit)
        print('gas evol.'.ljust(9)+' = '+(self.gasevol*'on'+(not self.gasevol)*'off').rjust(10))
        print('\n'+35*'-')
    
    def write_args(self):
        """
        Write out the simulation parameters to the file 'parameters.ini' in the
        folder specified in args.dir
        """
        import ConfigParser, os
        if not os.path.isdir(self.dir): os.mkdir(self.dir)
        parser = ConfigParser.ConfigParser()

        parser.add_section('parameters')
        for name,_ in self.varlist:
            parser.set('parameters', name, getattr(self, name))

        with open(self.dir+os.sep+'parameters.ini','w') as f:
            parser.write(f)

    def read_args(self):
        """
        Read in the simulation parameters from the file 'parameters.ini' in the
        folder specified in args.dir
        """
        import ConfigParser, os
        parser = ConfigParser.ConfigParser()
        parser.read(self.dir+os.sep+'parameters.ini')

        for name,t in self.varlist:
            if t is int:
                setattr(self, name, parser.getint('parameters', name))
            elif t is bool:
                setattr(self, name, parser.getboolean('parameters', name))
            elif t is float:
                setattr(self, name, parser.getfloat('parameters', name))

def lbp_solution(R,gamma,nu1,mstar,mdisk,RC0,time=0):
    """
    Calculate Lynden-Bell & Pringle self similar solution.
    All values need to be either given with astropy-units, or
    in as pure float arrays in cgs units.
    
    Arguments:
    ----------
    
    R : array
    : radius array
    
    gamma : float
    : viscosity exponent
    
    nu1 : float
    : viscosity at R[0]
    
    mstar : float
    : stellar mass
    
    mdisk : float
    : disk mass at t=0
    
    RC0 : float
    : critical radius at t=0
    
    Keywords:
    ---------
    
    time : float
    : physical "age" of the analytical solution 
    
    Output:
    -------
    sig_g,RC(t)
    
    sig_g : array
    : gas surface density, with or without unit, depending on input
    
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
    This is a wrapper for the two-population model `two_pop_model_run`, in which
    the disk profile is a self-similar solution.
    
    Arguments:
    ----------
    ARGS : instance of the input parameter object
    
    Keywords:
    ---------
    
    plot : bool
    :   whether or not to plot the default figures

    save : bool
    :   whether or not to write the data to disk
    
    Output:
    -------
    results : instance of the results object
    """
    import numpy as np
    import two_pop_model
    from matplotlib    import pyplot as plt
    from const         import AU, year, Grav, k_b, mu, m_p
    #
    # set parameters according to input
    #
    nr      = ARGS.nr
    nt      = ARGS.nt
    tmax    = ARGS.tmax
    n_a     = ARGS.na
    alpha   = ARGS.alpha
    d2g     = ARGS.d2g
    mstar   = ARGS.mstar
    tstar   = ARGS.tstar
    rstar   = ARGS.rstar
    rc      = ARGS.rc
    mdisk   = ARGS.mdisk
    rhos    = ARGS.rhos
    vfrag   = ARGS.vfrag
    a0      = ARGS.a0
    gamma   = ARGS.gamma
    edrift  = ARGS.edrift
    gasevol = ARGS.gasevol
    #
    # print setup
    #
    print(__doc__)
    print('\n'+35*'-')
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
    xi            = np.logspace(np.log10(0.05),np.log10(3e3),nri)*AU
    x             = 0.5*(xi[1:]+xi[:-1])
    timesteps     = np.logspace(4,np.log10(tmax/year),nt)*year
    T             = ( (0.05**0.25*tstar * (x /rstar)**-0.5)**4 + (7.)**4)**0.25
    #
    # set the initial surface density & velocity according Lynden-Bell & Pringle solution
    #
    alpha   = alpha*(x/x[0])**(gamma-1)
    om1     = np.sqrt(Grav*args.mstar/x[0]**3)
    cs1     = np.sqrt(k_b*T[0]/mu/m_p)
    nu1     = args.alpha*cs1**2/om1
    
    sigma_g,_ = lbp_solution(x, gamma, nu1, mstar, mdisk, rc)
    sigma_g = np.maximum(sigma_g,1e-100)
    sigma_d = sigma_g*d2g
    v_gas   = -3.0*alpha*k_b*T/mu/m_p/2./np.sqrt(Grav*mstar/x)*(1.+7./4.)
    #
    # call the model
    #
    [TI,SOLD,SOLG,VD,VG,v_0,v_1,a_dr,a_fr,a_df,a_t] = two_pop_model.two_pop_model_run(x,a0,timesteps,sigma_g,sigma_d,v_gas,T,alpha*np.ones(nr),mstar,vfrag,rhos,edrift,nogrowth=False,gasevol=gasevol)
    #
    # ================================
    # RECONSTRUCTING SIZE DISTRIBUTION
    # ================================
    #
    print('\n'+35*'-')
    if two_pop_model.distri_available:
        try:
            print('reconstructing size distribution')
            reconstruct_size_distribution = two_pop_model.reconstruct_size_distribution
            it = -1
            a  = np.logspace(np.log10(a0),np.log10(5*a_t.max()),n_a)
            sig_sol,_,_,_,_,_ = reconstruct_size_distribution(x,a,TI[it],SOLG[it],SOLD[-1],alpha*np.ones(nr),rhos,T,mstar,vfrag,a_0=a0)
        except Exception, _:
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
    res.T         = T
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
    
    if two_pop_model.distri_available:
        res.a       = a
        res.sig_sol = sig_sol
    if save: res.write()
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
    
    print(35*'-'+'\n')
    print('ALL DONE'.center(35))
    print('\n'+35*'-')
    return res
    
def run_test():
    """
    Test gas evolution: use small rc and large alpha
    """
    from const import AU
    Args       = args()
    Args.rc    = 20*AU
    Args.alpha = 1e-2
    res        = model_wrapper(Args)
    return res

def plot_test(res):
    """
    Plot the test result
    """
    from const import Grav, m_p, mu, k_b, year, AU
    import two_pop_run
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
    siga_0,_ = two_pop_run.lbp_solution(x,gamma,nu1,mstar,mdisk,rc)
    siga_1,_ = two_pop_run.lbp_solution(x,gamma,nu1,mstar,mdisk,rc,time=t)
    
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
    f.savefig('test.pdf')
    
if __name__=='__main__':
    import argparse
    import const as c
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
    ARGSIN.mdisk *= ARGSIN.mstar
    
    # convert to arguments object
    
    ARGS = args()
    for name,_ in ARGS.varlist:
        setattr(ARGS,name,getattr(ARGSIN, name))
        
    # call the wrapper
    
    model_wrapper(ARGS,save=True)