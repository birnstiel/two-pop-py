def num2tex(n,x=2,y=2):
    """
    num2tex
    This function turns a real number into a tex-string numbers >10^x and <10^-x
    are returned as e.g., $1.23 \times 10^{5}$, otherwise as e.g., $1234$.
    Unnecessary digit in the tex-string are removed

    Arguments:
    ----------
    
    n = number to be converted to tex string
    
    Keywords:
    ---------
    
    x : int
    :    threshold exponent
    
    y : int
    :    number of digits
    
    Example:
    --------
   
    >>> num2tex([3e3,3e4,3e5],5,1)
    '$3000.0$, $30000.0$, $3\times 10^{5}$'
    """
    from numpy import array,log10
    s=None;
    for i in array(n,ndmin=1):
        if i == 0:
            t=r'$0$';
        else:
            if log10(i)>x or log10(i)<-x:
                t = ('%2.'+str(y)+'e') % i
                t=t[0:t.find('e')]
                t=r'$'+t+r' \times 10^{%i}$' % round(log10(i/float(t)))
            else:
                t= ('%'+str(y)+'.'+str(y)+'f') %i
                t=r'$'+t+'$'
        #
        # some corrections
        #
        if y == 0:
            nz = ''
        else:
            nz = '.'+str(0).zfill(y)
        t=t.replace('1'+nz+' \times ','')
        t=t.replace(nz+' ','')
        #
        # we don't need 1\times 10^{x}, just 10^{x}
        #
        t=t.replace(r'$1\times','$')
        #
        # if there are more input numbers, attache them with a comma
        #
        if s != None:
            s=s+', '+t
        else:
            s=t
    return s

def loaddata_p3(dirname):
    '''
    Loads all the necessary data from the datafile
    '''
    import numpy as np
    import os
    from twopoppy.const import AU, year
    x         = np.loadtxt(dirname+os.sep+'x.dat') #radial grid
    t         = np.loadtxt(dirname+os.sep+'time.dat') #time steps
    # grain sizes
    a0 = 1e-5 # small population 
    a1 = np.loadtxt(dirname+os.sep+'a_t.dat') # large population
    adr = np.loadtxt(dirname+os.sep+'a_dr.dat') # drift limit
    afr = np.loadtxt(dirname+os.sep+'a_fr.dat') # fragmentation limit
    adf = np.loadtxt(dirname+os.sep+'a_df.dat') # drift fragmentation limit
    # column densities
    sig_g     = np.loadtxt(dirname+os.sep+'sigma_g.dat') #gas
    sig_d     = np.loadtxt(dirname+os.sep+'sigma_d.dat') #dust
    sig_p     = np.loadtxt(dirname+os.sep+'sigma_p.dat') #total plts
    #sig_i     = np.loadtxt(dirname+os.sep+'data'+os.sep+'sigma_i.dat') #icy plts
    #sig_r     = np.loadtxt(dirname+os.sep+'data'+os.sep+'sigma_r.dat') #rocky plts
    # temperature
    T      = np.loadtxt(dirname+os.sep+'T.dat') #power law
    #T_m       = np.loadtxt(dirname+os.sep+'data'+os.sep+'Tm.dat') # midplane temperature
    #T_s       = np.loadtxt(dirname+os.sep+'data'+os.sep+'Ts.dat') # surface temperature
    # accretion rate
    #v_gas     = np.loadtxt(dirname+os.sep+'data'+os.sep+'v_gas.dat') # gas velocity
    #M_dot = -2*pi*x*v_gas*sig_g
    return [x/AU,t/year,a0,a1,adr,adf,afr,sig_g,sig_d,sig_p,T]

def get_St(a,T,sigma,R,M_star,rho_s,Stokesregime=False):
    """
    Calculates the Stokesnumber in Epstein & Stokes regime 1
    USAGE: St = get_St(a,T,sigma,R,M_star)
    
    Arguments:
    ----------
    
    a : float/array
    :   grain size (array)
    
    T : float/array
    :   temperature in K
    
    sigma : float
    :    gas surface density [g/cm^2]
    
    M_star : float
    :   stellar mass [g]
    
    Keywords:
    ---------
    
    rho_s : float
    :   material density [g/cm^3]
     
    
    Stokesregime : bool
    :   if True, consider Stokes regime 1

    
    
    Output:
    -------
    
    Returns the Stokes number for the input grain sizes (array)
    """
    #
    # first some general constants
    #
    from twopoppy.const import k_b,mu,m_p,Grav,sig_h2
    from numpy import array,sqrt,pi,zeros,arange,newaxis

    
    a       = array(a,ndmin=1)
    sigma   = array(sigma,ndmin=1)
    cs      = sqrt(k_b*T/mu/m_p)
    omega_k = sqrt(Grav*M_star/R**3)
    H       = cs/omega_k
    n   = sigma/(sqrt(2.0*pi)*H*mu*m_p)
    lambd   = 0.5/(sig_h2*n)
    #
    # Epstein regime
    #
    #St = a[:,newaxis]*rho_s/sigma*stf # 'newaxis' is only needed if there is a size grid, but here a is a n_r dim array
    St = a*rho_s/sigma*pi/2.0
    #
    # Stokes regime
    #
    if Stokesregime:
        mask     = a/lambd>9./4.
        St[mask] = (2.0*pi/9.0*a**2*rho_s/(sigma*lambd))[mask]

    # return the lowest dimension necessary - squeeze reduces dimensions, the round
    # brackets strip a single scalar from the numpy array

    return St.squeeze()[()]

def loaddatap2(dirname):
    '''
    Loads all the necessary data from the datafile
    '''
    import numpy as np
    import os 
    from twopoppy.const import AU, year
    x         = np.loadtxt(dirname+os.sep+'data'+os.sep+'x.dat') #radial grid
    t         = np.loadtxt(dirname+os.sep+'data'+os.sep+'time.dat') #time steps
    # grain sizes
    a0 = 1e-5 # small population 
    a1 = np.loadtxt(dirname+os.sep+'data'+os.sep+'a_t.dat') # large population
    adr = np.loadtxt(dirname+os.sep+'data'+os.sep+'a_dr.dat') # drift limit
    afr = np.loadtxt(dirname+os.sep+'data'+os.sep+'a_fr.dat') # fragmentation limit
    adf = np.loadtxt(dirname+os.sep+'data'+os.sep+'a_df.dat') # drift fragmentation limit
    # column densities
    sig_g     = np.loadtxt(dirname+os.sep+'data'+os.sep+'sigma_g.dat') #gas
    sig_d     = np.loadtxt(dirname+os.sep+'data'+os.sep+'sigma_d.dat') #dust
    sig_p     = np.loadtxt(dirname+os.sep+'data'+os.sep+'sigma_p.dat') #total plts
    sig_i     = np.loadtxt(dirname+os.sep+'data'+os.sep+'sigma_i.dat') #icy plts
    sig_r     = np.loadtxt(dirname+os.sep+'data'+os.sep+'sigma_r.dat') #rocky plts
    # temperature
    T_pw      = np.loadtxt(dirname+os.sep+'data'+os.sep+'Tpw.dat') #power law
    T_m       = np.loadtxt(dirname+os.sep+'data'+os.sep+'Tm.dat') # midplane temperature
    T_s       = np.loadtxt(dirname+os.sep+'data'+os.sep+'Ts.dat') # surface temperature
    # accretion rate
    #v_gas     = np.loadtxt(dirname+os.sep+'data'+os.sep+'v_gas.dat') # gas velocity
    #M_dot = -2*pi*x*v_gas*sig_g
    return [x/AU,t/year,a0,a1,adr,adf,afr,sig_g,sig_d,sig_p,sig_i,sig_r,T_pw,T_m,T_s]