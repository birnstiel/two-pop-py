from .args import args


class results:
    nri          = None # noqa
    xi           = None # noqa
    x            = None # noqa
    timesteps    = None # noqa
    T            = None # noqa
    alpha        = None # noqa
    sigma_g      = None # noqa
    sigma_d      = None # noqa
    v_gas        = None # noqa
    v_dust       = None # noqa
    v_0          = None # noqa
    v_1          = None # noqa
    a_dr         = None # noqa
    a_fr         = None # noqa
    a_df         = None # noqa
    a_t          = None # noqa
    args         = None # noqa
    a            = None # noqa
    sig_sol      = None # noqa

    def write(self, dirname=None):
        """
        Export data to the specified folder.
        """
        import os
        import numpy as np

        if dirname is None:
            dirname = self.args.dir

        print('\n' + 35 * '-')
        print('writing results to {} ...'.format(dirname))
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        np.savetxt(dirname + os.sep + 'sigma_g.dat', self.sigma_g)                  # noqa
        np.savetxt(dirname + os.sep + 'sigma_d.dat', self.sigma_d)                  # noqa
        np.savetxt(dirname + os.sep + 'x.dat',       self.x)                        # noqa
        np.savetxt(dirname + os.sep + 'T.dat',       self.T)                        # noqa
        np.savetxt(dirname + os.sep + 'alpha.dat',   np.array(self.alpha, ndmin=1)) # noqa
        np.savetxt(dirname + os.sep + 'time.dat',    self.timesteps)                # noqa
        np.savetxt(dirname + os.sep + 'v_gas.dat',   self.v_gas)                    # noqa
        np.savetxt(dirname + os.sep + 'v_dust.dat',  self.v_dust)                   # noqa
        np.savetxt(dirname + os.sep + 'v_0.dat',     self.v_0)                      # noqa
        np.savetxt(dirname + os.sep + 'v_1.dat',     self.v_1)                      # noqa
        np.savetxt(dirname + os.sep + 'a_dr.dat',    self.a_dr)                     # noqa
        np.savetxt(dirname + os.sep + 'a_fr.dat',    self.a_fr)                     # noqa
        np.savetxt(dirname + os.sep + 'a_df.dat',    self.a_df)                     # noqa
        np.savetxt(dirname + os.sep + 'a_t.dat',     self.a_t)                      # noqa

        if self.a is not None and self.sig_sol is not None:
            np.savetxt(dirname + os.sep + 'a.dat', self.a)
            np.savetxt(dirname + os.sep + 'sigma_d_a.dat', self.sig_sol)

        self.args.write_args()

    def read(self, dirname=None):
        """
        Read results from the specified folder.
        """
        import os
        import numpy as np

        if dirname is None:
            if self.args.dir is not None:
                dirname = self.abs.dir
            else:
                dirname = 'data'

        print('\n' + 35 * '-')
        print('reading results from {} ...'.format(dirname))
        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        self.sigma_g = np.loadtxt(dirname + os.sep + 'sigma_g.dat')
        self.sigma_d = np.loadtxt(dirname + os.sep + 'sigma_d.dat')
        self.x = np.loadtxt(dirname + os.sep + 'x.dat')
        self.T = np.loadtxt(dirname + os.sep + 'T.dat')
        self.alpha = np.loadtxt(dirname + os.sep + 'alpha.dat')
        self.timesteps = np.loadtxt(dirname + os.sep + 'time.dat')
        self.v_gas = np.loadtxt(dirname + os.sep + 'v_gas.dat')
        self.v_0 = np.loadtxt(dirname + os.sep + 'v_0.dat')
        self.v_1 = np.loadtxt(dirname + os.sep + 'v_1.dat')
        self.a_dr = np.loadtxt(dirname + os.sep + 'a_dr.dat')
        self.a_fr = np.loadtxt(dirname + os.sep + 'a_fr.dat')
        self.a_df = np.loadtxt(dirname + os.sep + 'a_df.dat')
        self.a_t = np.loadtxt(dirname + os.sep + 'a_t.dat')

        if os.path.isfile(dirname + os.sep + 'a.dat'):
            self.a = np.loadtxt(dirname + os.sep + 'a.dat')
        if os.path.isfile(dirname + os.sep + 'sigma_d_a.dat'):
            self.sig_sol = np.loadtxt(dirname + os.sep + 'sigma_d_a.dat')

        self.args = args()
        self.args.dir = dirname
        self.args.read_args()
