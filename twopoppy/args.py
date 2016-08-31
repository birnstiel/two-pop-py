class args:
    import const as _c
    
    # names of all parameters

    varlist = [ ['nr',        int],
                ['nt',        int],
                ['tmax',    float],
                ['alpha',   float],
                ['d2g',     float],
                ['mstar',   float],
                ['tstar',   float],
                ['rstar',   float],
                ['rc',      float],
                ['mdisk',   float],
                ['rhos',    float],
                ['vfrag',   float],
                ['a0',      float],
                ['gamma',   float],
                ['edrift',  float],
                ['T',       float],
                ['gasevol',  bool],
                ['tempevol', bool],
                ['starevol', bool],
            ]
    
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

    gasevol  = True
    tempevol = False
    starevol = False
    T        = None
    dir      = 'data'
    
    def __init__(self,**kwargs):
        """
        Initialize arguments. Simulation parameters can be given as keywords.
        To list all parameters, call `print_args()`. All quantities need to be
        given in CGS units, even though they might be printed by `print_args`
        using other units.
        """
        import warnings
        for k,v in kwargs.iteritems():
            if hasattr(self,k):
                setattr(self,k,v)
            else:
                warnings.warn("No such argument")
    
    def __str__(self):
        """
        String representation of arguments
        """
        from const import year, M_sun, R_sun, AU
        s = ''
        s+=35*'-'+'\n'
        
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
            value     = getattr(self, n)
            isarray   = hasattr(value, '__len__')
            if isarray:
                s+=n.ljust(17)+' = '+'[{:3.2g} ... {:3.2g}]'.format(conv*value[0],conv*value[-1]).rjust(15)+' '+unit+'\n'
            else:
                s+=n.ljust(17)+' = '+'{:3.2g}'.format(conv*value).rjust(15)+' '+unit+'\n'
            
        # print other arguments
            
        s += 'Gas         evol.'.ljust(17)+' = '+(self.gasevol *'on'+(not self.gasevol )*'off').rjust(15)+'\n'
        s += 'Temperature evol.'.ljust(17)+' = '+(self.tempevol*'on'+(not self.tempevol)*'off').rjust(15)+'\n'
        s += 'Stellar     evol.'.ljust(17)+' = '+(self.starevol*'on'+(not self.starevol)*'off').rjust(15)+'\n'
        
        if self.T is None:
            s += 'Temperature'.ljust(17)+' = '+'None'.rjust(15)
        elif hasattr(self.T,'__call__'):
            s += 'Temperature'.ljust(17)+' = '+'function'.rjust(15)
        else:
            s += 'Temperature'.ljust(17)+' = '+'{}'.format(self.T).rjust(15)
        
        s += '\n'+35*'-'+'\n'
        return s
        
    def print_args(self):
        print(self.__str__())
    
    def write_args(self,fname=None):
        """
        Write out the simulation parameters to the file 'parameters.ini' in the
        folder specified in args.dir or to fname if that is not None
        """
        import configobj, os
        if fname is None:
            fname  = os.path.abspath(os.path.expanduser(fname))
            folder = os.path.dirname(fname)
            fname  = os.path.basename(fname)
        else:
            folder = self.dir
            fname  = 'parameters.ini'
            
        if not os.path.isdir(folder): os.mkdir(folder)
        parser = configobj.ConfigObj()
        parser.filename = os.path.join(folder,fname)

        for name,_ in self.varlist: parser[name] = getattr(self, name)

        parser.write()

    def read_args(self):
        """
        Read in the simulation parameters from the file 'parameters.ini' in the
        folder specified in args.dir
        """
        import configobj, os, numpy
        parser = configobj.ConfigObj(self.dir+os.sep+'parameters.ini')
        
        varlist = {v[0]:v[1] for v in self.varlist}
        
        for name,val in parser.iteritems():
            if name not in varlist:
                print('Unknown Parameter:{}'.format(name))
                continue
            
            t = varlist[name]
            
            # process ints, bools, floats and lists of them
            
            if t in [int,bool,float]:
                if type(val) is list:
                    # lists
                    setattr(self, name, [t(v) for v in val])
                elif '[' in val:
                    # numpy arrays
                    val = val.replace('[','').replace(']','').replace('\n',' ')
                    val = [t(v) for v in val.split()]
                    setattr(self, name, numpy.array(val))
                else:
                    # plain values
                    setattr(self, name, t(val))
            else:
                # stings and nones
                if val=='None':
                    val = None
                else:
                    setattr(self,name,val)