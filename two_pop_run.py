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
    			['edrift', float]]
    
    # set default values
    
    nr      = 200
    nt      = 100
    na      = 150
    tmax    = 1e6
    alpha   = 1e-3
    d2g     = 1e-2
    mstar   = 0.7
    tstar   = 4010.
    rstar   = 1.806
    rc      = 200
    mdisk   = 0.1
    rhos    = 1.156
    vfrag   = 1000
    a0      = 1e-5
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
    from const         import AU, year, Grav, M_sun, k_b, mu, m_p, R_sun, pi
    #
    # set parameters according to input
    #
    nr      = ARGS.nr
    nt      = ARGS.nt
    tmax    = ARGS.tmax*year
    n_a     = ARGS.na
    alpha   = ARGS.alpha
    d2g     = ARGS.d2g
    mstar   = ARGS.mstar*M_sun
    tstar   = ARGS.tstar
    rstar   = ARGS.rstar*R_sun
    rc      = ARGS.rc*AU
    mdisk   = ARGS.mdisk*mstar
    rhos    = ARGS.rhos
    vfrag   = ARGS.vfrag
    a0      = ARGS.a0
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
    T             = ( (0.05**0.25*tstar * (x /rstar)**-0.5)**4 + 1e4)**0.25
    #
    # set the initial surface density & velocity
    #
    sigma_g     = np.maximum(mdisk/(2*pi*rc**2)*(rc/x)*np.exp(-x/rc),1e-100)
    sigma_d     = sigma_g*d2g
    v_gas       = -3.0*alpha*k_b*T/mu/m_p/2./np.sqrt(Grav*mstar/x)*(1.+7./4.)
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
        print('reconstructing size distribution')
        reconstruct_size_distribution = two_pop_model.reconstruct_size_distribution
        it = -1
        a  = np.logspace(np.log10(a0),np.log10(5*a_t.max()),n_a)
        sig_sol,_,_,_,_,_ = reconstruct_size_distribution(x,a,TI[it],SOLG[it],SOLD[-1],alpha*np.ones(nr),rhos,T,mstar,vfrag,a_0=a0)
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
    
def test():
    from const import AU
    import matplotlib.pyplot as plt 
    Args = args()
    res  = model_wrapper(Args)
    
    x     = res.x
    sig_g = res.sigma_g[-1]
    t     = res.timesteps[-1]
    temp  = res.T[-1]
    alpha = Args.alpha
    
    _,ax = plt.subplots()
    ax.loglog(x/AU,sig_g)
    
if __name__=='__main__':
    import argparse
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
    PARSER.add_argument('-edrift',help='drift fudge factor',                   type=float, default=1.0)
    PARSER.add_argument('-dir',   help='output directory default: data/',      type=str,   default='data')
    
    PARSER.add_argument('-p',               help='produce plots if possible',  action='store_true')
    PARSER.add_argument('-g','--gasevol',   help='turn *off* gas evolution',   action='store_false')
    ARGSIN = PARSER.parse_args()
    
    # convert to arguments object
    
    ARGS = args()
    for name,_ in ARGS.varlist:
        setattr(ARGS,name,getattr(ARGSIN, name))
        
    # call the wrapper
    
    model_wrapper(ARGS,save=True)