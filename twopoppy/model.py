def run(x,a_0,time,sig_g,sig_d,v_gas,T,alpha,m_star,V_FRAG,RHO_S,E_drift,E_stick=1.,nogrowth=False,gasevol=True):
    """
    This function evolves the two population model (all model settings
    are stored in velocity). It returns the important parameters of
    the model.
    
    
    Arguments:
    ----------
    x : array
        radial grid (nr)                [cm]

    a_0 : float
        monomer size                    [cm]

    time : array
        time of snapshots (nt)          [s]

    sig_g : array
        gas surface density (nr)        [g cm^-2]

    sig_d : array
        dust surface density (nr)       [g cm^-2]

    v_gas : array
        gas velocity (nr)               [cm/s]

    T : array
        temperature array (nr)/function [K]

    alpha : array
        turbulence parameter (nr)       [-]

    m_star : float
        stellar mass (nt)               [g]

    V_FRAG : float
        fragmentation velocity          [cm s^-1]

    RHO_S : float
        internal density of the dust    [g cm^-3]

    E_drift : float
        drift efficiency                [-]
        
    E_stick : float
        sticking probability            [-]

        
    Keywords:
    ---------
    
    nogrowth : bool
        true: particle size fixed to a0 [False]

    gasevol : bool
        turn gas evolution on/off       [True]

    
    Returns:
    ---------
    
    time : array
        snapshot time            (nt)          [s]

    solution_d : array
        dust surface density     (nt,nr)       [g cm^-2]

    solution_g : array
        dust surface density     (nt,nr)       [g cm^-2]

    v_bar : array
        dust velocity            (nt,nr)       [cm s^-1]

    v_gas : array
        gas velocity             (nt,nr)       [cm s^-1]

    v_0 : array
        small dust velocity      (nt,nr)       [cm s^-1]

    v_1 : array
        large dust velocity      (nt,nr)       [cm s^-1]

    a_dr : array
        drift size limit         (nt,nr)       [cm]

    a_fr : array
        fragmentation size limit (nt,nr)       [cm]

    a_df : array
        drift-ind. frag. limit   (nt,nr)       [cm]

    a_t : array
        the time dependent limit (nt,nr)       [cm]
    
    Note:
    -----
    
    the temperature (and also alpha) can be an array (with nr elements) or it
    can be a function that function is always just called with two arguments r
    and locals(). This allows the user to access local information like surface
    density if necessary (nonsense-example):

        def T(x,locals_):
            return 10*(x/x[-1])**-1.5 * locals_['sig_g']/locals_['sig_g'][-1]
    
    
    but still keeps things simple enough to do something like this:
    
        def T(x,locals_):
            return 200*(x/AU)**-1
    """
    from numpy     import ones,zeros,Inf,maximum,minimum,sqrt,where
    from .const     import year,Grav,k_b,mu,m_p
    import sys
    #
    # some setup
    #
    n_r    = len(x)
    n_t    = len(time)
    g      = ones(n_r)
    K      = zeros(n_r)
    L      = zeros(n_r)
    flim   = ones(n_r)
    A0     = zeros(n_r)
    B0     = zeros(n_r)
    C0     = zeros(n_r)
    D0     = zeros(n_r)
    #
    # setup
    #
    t               = time[0]
    solution_d      = zeros([n_t,n_r])
    solution_d[0,:] = sig_d
    solution_g      = zeros([n_t,n_r])
    solution_g[0,:] = sig_g
    vgas            = zeros([n_t,n_r])
    vgas[0,:]       = v_gas
    v_bar           = zeros([n_t,n_r])
    Diff            = zeros([n_t,n_r])
    v_0             = zeros([n_t,n_r])
    v_1             = zeros([n_t,n_r])
    a_t             = zeros([n_t,n_r])
    a_df            = zeros([n_t,n_r])
    a_fr            = zeros([n_t,n_r])
    a_dr            = zeros([n_t,n_r])
    Tout            = zeros([n_t,n_r])
    alphaout        = zeros([n_t,n_r])
    u_in            = solution_d[0,:]*x
    it_old          = 1
    snap_count      = 0
    progress_bar(round((it_old-1)/(n_t-1)*100),'toy model running')
    
    # T can be either an array or a specific function.
    # in either case, we define a function that returns the temperature
    
    if hasattr(T, '__call__'):
        Tfunc = T
    else:
        def Tfunc(x,locals_):
            return T
            
    # alpha can be either an array or a specific function.
    # in either case, we define a function that returns it
    
    if hasattr(alpha, '__call__'):
        alpha_func = alpha
    else:
        def alpha_func(x,locals_):
            return alpha

    #
    # save the velocity which will be used
    #
    res  = get_velocity(t,solution_d[0,:],x,sig_g,v_gas,Tfunc(x,locals()),alpha_func(x,locals()),m_star,a_0,V_FRAG,RHO_S,E_drift,E_stick=E_stick,nogrowth=nogrowth)
    v_bar[0,:]    = res[0]
    Diff[0,:]     = res[1]
    v_0[0,:]      = res[2]
    v_1[0,:]      = res[3]
    a_t[0,:]      = res[4]
    a_df[0,:]     = res[5]
    a_fr[0,:]     = res[6]
    a_dr[0,:]     = res[7]
    Tout[0,:]     = Tfunc(x,locals())
    alphaout[0,:] = alpha_func(x,locals())
    #
    # the loop
    #
    dt = Inf
    while t<time[-1]:
        #
        # set the time step
        #
        dt = min(dt*10,time[it_old]-t)
        if t != 0.0: dt = min(dt,t/200.0)
        if dt==0:
            print('ERROR:')
            print('t      = %g years'%(t/year))
            print('it_old = %g'%it_old)
            print('dt = 0')
            sys.exit(1)
            
        # update the temperature and alpha
            
        _T     = Tfunc(x,locals())
        _alpha = alpha_func(x,locals())
            
        
        # calculate the velocity
        
        res    = get_velocity(t,u_in/x,x,sig_g,v_gas,_T,_alpha,m_star,a_0,V_FRAG,RHO_S,E_drift,E_stick=E_stick,nogrowth=nogrowth)
        v      = res[0]
        D      = res[1]
        v[0]   = v[1]
        D[0]   = D[1]
        D[-2:] = 0
        v[-2:] = 0
        #
        # set up the equation
        #
        h    = sig_g*x
        #
        # do the update
        #
        u_dust = impl_donorcell_adv_diff_delta(n_r,x,D,v,g,h,K,L,flim,u_in,dt,1,1,0,0,0,0,1,A0,B0,C0,D0)
        mask = abs(u_dust[2:-1]/u_in[2:-1]-1)>0.05
        #
        # try variable time step
        #
        while any(u_dust[2:-1][mask]/x[2:-1][mask]>=1e-30):
            dt = dt/10.
            if dt<1.0 and snap_count>0:
                print('ERROR: time step got too short')
                sys.exit(1)
            u_dust = impl_donorcell_adv_diff_delta(n_r,x,D,v,g,h,K,L,flim,u_in,dt,1,1,0,0,0,0,1,A0,B0,C0,D0)
            mask = abs(u_dust[2:-1]/u_in[2:-1]-1)>0.3
        #
        # update
        #
        u_in  = u_dust[:]
        t     = t + dt
        #
        # update the gas
        #
        if gasevol:
            nu_gas     = _alpha * k_b*_T/mu/m_p * sqrt(x**3/Grav/m_star)
            u_gas_old  = sig_g*x
            u_gas      = u_gas_old[:]
            v_gas      = zeros(n_r)
            D_gas      = 3.0*sqrt(x)
            g_gas      = nu_gas/sqrt(x)
            h_gas      = ones(n_r)
            K_gas      = zeros(n_r)
            L_gas      = zeros(n_r)
            p_L        = -(x[1]-x[0])*h_gas[1]/(x[1]*g_gas[1])
            q_L        = 1./x[0]-1./x[1]*g_gas[0]/g_gas[1]*h_gas[1]/h_gas[0]
            r_L        = 0.0
            u_gas      =  impl_donorcell_adv_diff_delta(n_r,x,D_gas,v_gas,g_gas,h_gas,K_gas,L_gas,flim,u_gas,dt,p_L,0.0,q_L,1.0,r_L,1e-100*x[n_r-1],1,A0,B0,C0,D0)
            sig_g      = u_gas/x
            sig_g      = maximum(sig_g,1e-100)
            #
            # now get the gas velocities from the exact fluxes
            #
            u_flux          = zeros(n_r)
            u_flux[1:n_r+1] = - flim[1:n_r+1]*0.25*(D_gas[1:]+D_gas[:-1]) * (h_gas[1:]+h_gas[:-1]) * (g_gas[1:]/h_gas[1]*u_gas[1:]-g_gas[:-1]/h_gas[:-1]*u_gas[:-1]) / (x[1:]-x[:-1])
            mask            = u_flux>0.0
            imask           = u_flux<=0.0
            v_gas[mask]     = u_flux[mask] /u_gas[maximum(0,    where(mask) [0]-1)]
            v_gas[imask]    = u_flux[imask]/u_gas[minimum(n_r-1,where(imask)[0]+1)]
        #
        # find out if we reached a snapshot
        #
        if t>=time[it_old]:
            #
            # one more step completed
            #
            it_old     = it_old + 1
            snap_count = snap_count + 1
            #
            # notify
            #
            progress_bar(round((it_old-1.)/(n_t)*100.),'toy model running')
            #
            # save the data
            #
            solution_d[snap_count,:] = u_dust/x
            solution_g[snap_count,:] = sig_g
            v_bar[snap_count,:]      = v
            vgas[snap_count,:]       = v_gas
            Diff[snap_count,:]       = D
            #
            # store the rest
            #
            v_0[snap_count,:]      = res[2]
            v_1[snap_count,:]      = res[3]
            a_t[snap_count,:]      = res[4]
            a_df[snap_count,:]     = res[5]
            a_fr[snap_count,:]     = res[6]
            a_dr[snap_count,:]     = res[7]
            Tout[snap_count,:]     = _T
            alphaout[snap_count,:] = _alpha

    progress_bar(100.,'toy model running')

    return time,solution_d,solution_g,v_bar,vgas,v_0,v_1,a_dr,a_fr,a_df,a_t,Tout,alphaout


def get_velocity(t,sigma_d_t,x,sigma_g,v_gas,T,alpha,m_star,a_0,V_FRAG,RHO_S,E_drift,E_stick=1.,nogrowth=False):
    """
    This model takes a snapshot of temperature, gas surface density and so on
    and calculates values of the representative sizes and velocities which are
    used in the two population model.
    
   
    Arguments:
    ----------
    t : float
        time at which to calculate the values  [s]
        
    sigma_d_t : array-like
        current dust surface density array (nr)[g cm^-2]
        
    x : array-like
        nr radial grid points (nr)             [cm]
                
    sigma_g : array-like
        gas surface density (nr)               [g cm^-2]
        
    v_gas : array-like
        gas radial velocity (nr)               [cm s^-1]
        
    T : array-like
        temperature (nr)                       [K]
        
    alpha : array-like
        turbulence parameter (nr)              [-]
        
    m_star : float
        stellar mass                           [g]
        
    a_0 : float
        monomer size                           [cm]
        
    V_FRAG : float
        fragmentation velocity                 [cm s^-1]
        
    RHO_S : float
        dust internal density                  [g cm^-3]
        
    E_drift : float
        drift efficiency                       [-]
    
    Keywords:
    ---------
    
    E_stick : float
        sticking probability                   [-]
    
    nogrowth : bool
        wether a fixed particle size is used   [False]
    
    Returns:
    --------
    v_bar : array
        the mass averaged velocity (nr)         [cm s^-1]
        
    D : array
        t-interpolated diffusivity (nr)         [cm^2 s^-1]

    v_0 : array
        t-interpolated vel. of small dust (nr)  [cm s^-1]
        
    v_1 : array
        t-interpolated vel. of large dust (nr)  [cm s^-1]
        
    a_max_t : array
        maximum grain size (nr)                 [cm]

    a_df : array
        the fragmentation-by-drift limit (nr)   [cm]
        
    a_fr : array
        the fragmentation limit (nr)            [cm]
        
    a_dr : the drift limit (nr)                 [cm]
    """
    fudge_fr = 0.37
    fudge_dr = 0.55
    #
    # set some constants
    #
    from .const import k_b,mu,m_p,Grav
    from numpy import ones,zeros,maximum,minimum,sqrt,array,exp,invert,pi
    n_r = len(x)
    #
    # calculate the pressure power-law index
    #
    P              = sigma_g  * sqrt(Grav*m_star/x**3) * sqrt(k_b*T/mu/m_p)
    gamma          = zeros(n_r)
    gamma[1:n_r-1] = x[1:n_r-1]/P[1:n_r-1]*(P[2:n_r]-P[0:n_r-2])/(x[2:n_r]-x[0:n_r-2])
    gamma[0]       = gamma[1]
    gamma[-1]      = gamma[-2]
    #
    # calculate the sizes
    #
    o_k         = sqrt(Grav*m_star/x**3)
    if nogrowth:
        mask        = ones(n_r)==1
        a_max       = a_0*ones(n_r)
        a_max_t     = a_max
        a_max_t_out = a_max
        a_fr        = a_max
        a_dr        = a_max
        a_df        = a_max
    else:
        a_fr  = fudge_fr*2*sigma_g*V_FRAG**2/(3*pi*alpha*RHO_S*k_b*T/mu/m_p)
        a_dr  = E_stick*fudge_dr/E_drift*2/pi*sigma_d_t/RHO_S*x**2*(Grav*m_star/x**3)/(abs(gamma)*(k_b*T/mu/m_p))
        N     = 0.5
        a_df  = fudge_fr*2*sigma_g/(RHO_S*pi)*V_FRAG*sqrt(Grav*m_star/x)/(abs(gamma)*k_b*T/mu/m_p*(1-N))
        a_max = maximum(a_0*ones(n_r),minimum(a_dr,a_fr))
        ###
        # EXPERIMENTAL: inlcude a_df as upper limit
        a_max     = maximum(a_0*ones(n_r),minimum(a_df,a_max))
        a_max_out = minimum(a_df,a_max)
        #mask      = all([a_dr<a_fr,a_dr<a_df],0)
        mask = array([adr<afr and adr<adf for adr,afr,adf in zip(a_dr,a_fr,a_df)])
        ###
        #
        # calculate the growth time scale and thus a_1(t)
        #
        tau_grow    = sigma_g/maximum(1e-100,E_stick*sigma_d_t*o_k)
        a_max_t     = minimum(a_max,a_0*exp(minimum(709.0,t/tau_grow)))
        a_max_t_out = minimum(a_max_out,a_0*exp(minimum(709.0,t/tau_grow)))
    #
    # calculate the Stokes number of the particles
    #
    St_0 = a_0     * RHO_S/sigma_g*pi/2
    St_1 = a_max_t * RHO_S/sigma_g*pi/2
    #
    # calculate the velocities of the two populations:
    # First: gas velocity
    #
    v_0 = v_gas/(1+St_0**2)
    v_1 = v_gas/(1+St_1**2)
    #
    # Second: drift velocity
    #
    v_dr = k_b*T/mu/m_p/(2*o_k*x)*gamma
    #
    # level of at the peak position
    #
    v_0 = v_0 + 2/(St_0+1/St_0)*v_dr
    v_1 = v_1 + 2/(St_1+1/St_1)*v_dr
    #
    # set the mass distribution ratios
    #
    f_m = 0.75*invert(mask)+0.97*mask
    #
    # calculate the mass weighted transport velocity
    #
    v_bar = v_0*(1-f_m) + v_1*f_m
    #
    # calculate the diffusivity
    #
    D = alpha * k_b*T/mu/m_p/o_k
    
    return [v_bar,D,v_0,v_1,a_max_t_out,a_df,a_fr,a_dr]


def impl_donorcell_adv_diff_delta(n_x,x,Diff,v,g,h,K,L,flim,u_in,dt,pl,pr,ql,qr,rl,rr,coagulation_method,A,B,C,D):
    """
    Implicit donor cell advection-diffusion scheme with piecewise constant values
    
    NOTE: The cell centers can be arbitrarily placed - the interfaces are assumed
    to be in the middle of the "centers", which makes all interface values
    just the arithmetic mean of the center values.
    
        Perform one time step for the following PDE:
    
           du    d  /    \    d  /              d  /       u   \ \
           -- + -- | u v | - -- | h(x) Diff(x) -- | g(x) ----  | | = K + L u
           dt   dx \    /    dx \              dx \      h(x) / /
    
        with boundary conditions
    
            dgu/h |            |
          p ----- |      + q u |       = r
             dx   |x=xbc       |x=xbc
             
    Arguments:
    ----------
    n_x : int
        number of grid points

    x : array-like
        the grid

    Diff : array-like
        value of Diff @ cell center

    v : array-like
        the values for v @ interface (array[i] = value @ i-1/2)

    g : array-like
        the values for g(x)

    h : array-like
        the values for h(x)

    K : array-like
        the values for K(x)

    L : array-like
        the values for L(x)

    flim : array-like
        diffusion flux limiting factor at interfaces

    u : array-like
        the current values of u(x)

    dt : float
        the time step

    
    Output:
    -------
    
    u : array-like
        the updated values of u(x) after timestep dt

    """
    from numpy import zeros
    D05=zeros(n_x)
    h05=zeros(n_x)
    rhs=zeros(n_x)
    #
    # calculate the arrays at the interfaces
    #
    for i in range(1,n_x):
        D05[i] = flim[i] * 0.5 * (Diff[i-1] + Diff[i])
        h05[i] = 0.5 * (h[i-1] + h[i])
    #
    # calculate the entries of the tridiagonal matrix
    #
    for i in range(1,n_x-1):
        vol = 0.5*(x[i+1]-x[i-1])
        A[i] = -dt/vol *  \
            ( \
            + max(0.,v[i])  \
            + D05[i] * h05[i] * g[i-1] / (  (x[i]-x[i-1]) * h[i-1]  ) \
            )
        B[i] = 1. - dt*L[i] + dt/vol * \
            ( \
            + max(0.,v[i+1])   \
            - min(0.,v[i])  \
            + D05[i+1] * h05[i+1] * g[i]   / (  (x[i+1]-x[i]) * h[i]    ) \
            + D05[i]   * h05[i]   * g[i]   / (  (x[i]-x[i-1]) * h[i]    ) \
            )
        C[i] = dt/vol *  \
            ( \
            + min(0.,v[i+1])  \
            - D05[i+1] * h05[i+1]  * g[i+1] / (  (x[i+1]-x[i]) * h[i+1]  ) \
            )
        D[i] = -dt * K[i]
    #
    # boundary Conditions
    #
    A[0]   = 0.
    B[0]   = ql - pl*g[0] / (h[0]*(x[1]-x[0]))
    C[0]   =      pl*g[1] / (h[1]*(x[1]-x[0]))
    D[0]   = u_in[0]-rl
    
    A[-1] =    - pr*g[-2] / (h[-2]*(x[-1]-x[-2]))
    B[-1] = qr + pr*g[-1]  / (h[-1]*(x[-1]-x[-2]))
    C[-1] = 0.
    D[-1] = u_in[-1]-rr
    #
    # if coagulation_method==2,
    #  then we change the arrays and
    #  give them back to the calling routine
    # otherwise, we solve the equation
    #
    if  coagulation_method==2:  
        A = A/dt           ##ok<NASGU>
        B = (B - 1.)/dt   ##ok<NASGU>
        C = C/dt           ##ok<NASGU>
        D = D/dt           ##ok<NASGU>
    else:
        #
        # the old way
        #
        #rhs = u - D
    
        #
        # the delta-way
        #
        for i in range(1,n_x-1):
            rhs[i] = u_in[i] - D[i] - (A[i]*u_in[i-1]+B[i]*u_in[i]+C[i]*u_in[i+1])
        rhs[0]   = rl - (               B[0] *u_in[0]  + C[0]*u_in[1])
        rhs[-1]  = rr - (A[-1]*u_in[-2]+B[-1]*u_in[-1]               )
        #
        # solve for u2
        #
        u2=tridag(A,B,C,rhs,n_x);
        #
        # update u
        #u = u2   # old way
        #
        u_out = u_in+u2 # delta way
    
    return u_out

def tridag(a,b,c,r,n):
    """
    Solves a tridiagnoal matrix equation
    
        M * u  =  r
    
    where M is tridiagonal, and u and r are vectors of length n.
    
    Arguments:
    ----------
    
    a : array
        lower diagonal entries
    
    b : array
        diagonal entries
    
    c : array
        upper diagonal entries
    
    r : array
        right hand side vector
    
    n : int
        size of the vectors
    
    Returns:
    --------
    
    u : array
        solution vector
    """
    import numpy as np
    
    gam = np.zeros(n)
    u   = np.zeros(n)
    
    if b[0]==0.:
        raise ValueError('tridag: rewrite equations')
    
    bet = b[0]
    
    u[0]=r[0]/bet
    
    for j in np.arange(1,n):
        gam[j] = c[j-1]/bet
        bet    = b[j]-a[j]*gam[j]
        
        if bet==0:
            raise ValueError('tridag failed')
        u[j]   = (r[j]-a[j]*u[j-1])/bet
    
    for j in np.arange(n-2,-1,-1):
        u[j] = u[j]-gam[j+1]*u[j+1]
    return u

def progress_bar(perc,text=''):
    """
    This is a very simple progress bar which displays the given
    percentage of completion on the command line, overwriting its
    previous output.

    Arguments:
    ----------
    
    perc : float
        The percentage of completion, should be
        between 0 and 100. Only 100.0 finishes with the
        word "Done!".
         
    Keywords:
    ---------
    
    text : str
        Possible text for describing the running process.
    
    Example:
    --------
    
    >>> import time
    >>> for i in linspace(0,100,1000):
    >>>     progress_bar(i,text='Waiting')
    >>>     time.sleep(0.005)
    """
    import sys 
    
    if text!='': text = text+' ... '
    
    if perc==100.0:
        sys.stdout.write('\r'+text+'Done!\n')
        sys.stdout.flush()
    else:
        sys.stdout.write('\r'+text+'%d %%'%round(perc))
        sys.stdout.flush()


distri_available = False
_helperfiles     = ['distribution_reconstruction.py','aux_functions.py']
_gitpath         = 'https://raw.githubusercontent.com/birnstiel/Birnstiel2015_scripts/master/'


try:
    from distribution_reconstruction import reconstruct_size_distribution
    distri_available=True
except (ImportError, SyntaxError):
    import os             as _os
    import inspect        as _inspect
    import ssl            as _ssl
    import warnings       as _warnings
    import traceback      as _traceback
    import urllib.request as _request
    
    print('could not import distribution_reconstruction.py, will download it and retry')
    
    _module_path = _os.path.dirname(_os.path.abspath(_inspect.getfile(_inspect.currentframe())))
    for _file in _helperfiles:
        try:
            ctx = _ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = _ssl.CERT_NONE
            open(_module_path+_os.sep+_file, 'wb').write(_request.urlopen(_gitpath+_file,context=ctx).read())
        except:
            _warnings.warn('Could not download {}'.format(_file))
            for s in _traceback.format_exc().split('\n'):  print(4*' '+s)
            
    try:
        from distribution_reconstruction import reconstruct_size_distribution
        distri_available=True
        print('success!')
    except (ImportError,SyntaxError):
        for _file in _helperfiles: _os.system('wget --no-check-certificate '+_gitpath+_file+' -O '+_module_path+_os.sep+_file)
        try:
            from distribution_reconstruction import reconstruct_size_distribution
            distri_available=True
            print('success!')
        except (ImportError,SyntaxError):
            distri_available=False
            print('Could not import distribution_reconstruction.py')