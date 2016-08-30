from .args import args

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
        import model
        
        if dirname is None: dirname = self.args.dir
        
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
        
        if model.distri_available:
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
        self.args.read(dirname=dirname)