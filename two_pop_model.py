def two_pop_model_run(x,a_0,time,sig_g,sig_d,v_gas,T,alpha,m_star,V_FRAG,RHO_S,E_drift,nogrowth=False,gasevol=True):
    """
    This function evolves the two population model (all model settings
    are stored in two_pop_velocity). It returns the important parameters of
    the model.
    
    USAGE:
    [time,solution_d,solution_g,v_bar,v_0,v_1,a_dr,a_fr,a_df,a_t]  =  two_pop_model_run_ic ...
    (x,a_0,time,sig_g,sig_d,v_gas,T,alpha,m_star,V_FRAG,RHO_S,E_drift):
    
    WHERE:
        x               = radial grid (nr)                [cm]
        a_0             = monomer size                    [cm]
        time            = time of snapshots (nt)          [s]
        sig_g           = gas  surface density (nr)       [g cm^-2]
        sig_d           = dust surface density (nr)       [g cm^-2]
        v_gas           = gas velocity (nr)               [cm/s]
        T               = temperature grid (nr)           [K]
        alpha           = turbulence parameter (nr)       [-]
        m_star          = stellar mass (nt)               [g]
        V_FRAG          = fragmentation velocity          [cm s^-1]
        RHO_S           = internal density of the dust    [g cm^-3]
        E_drift         = drift efficiency                [-]
        
    KEYWORDS:
        nogrowth        = true: particle size fixed to a0 [False]
        gasevol         = turn gas evolution on/off       [True]
    
    RETURNS:
    
        time       = snap shot time           (nt,nr)       [s]
        solution_d = dust surface density     (nt,nr)       [g cm^-2]
        solution_g = dust surface density     (nt,nr)       [g cm^-2]
        v_bar      = dust velocity            (nt,nr)       [cm s^-1]
        v_gas      = gas velocity             (nt,nr)       [cm s^-1]
        v_0        = small dust velocity      (nt,nr)       [cm s^-1]
        v_1        = large dust velocity      (nt,nr)       [cm s^-1]
        a_dr       = drift size limit         (nt,nr)       [cm]
        a_fr       = fragmentation size limit (nt,nr)       [cm]
        a_df       = drift-ind. frag. limit   (nt,nr)       [cm]
        a_t        = the time dependent limit (nt,nr)       [cm]
    """
    from numpy     import ones,zeros,Inf,maximum,minimum,sqrt,where
    from const     import year,Grav,k_b,mu,m_p
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
    u_in            = solution_d[0,:]*x
    it_old          = 1
    snap_count      = 0
    progress_bar(round((it_old-1)/(n_t-1)*100),'toy model running')
    #
    # save the velocity which will be used
    #
    res  = two_pop_velocity(t,solution_d[0,:],x,sig_g,v_gas,T,alpha,m_star,a_0,V_FRAG,RHO_S,E_drift,nogrowth=nogrowth)
    v_bar[0,:] = res[0]
    Diff[0,:]  = res[1]
    v_0[0,:]   = res[2]
    v_1[0,:]   = res[3]
    a_t[0,:]   = res[4]
    a_df[0,:]  = res[5]
    a_fr[0,:]  = res[6]
    a_dr[0,:]  = res[7]
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
        #
        # calculate the velocity
        #
        res    = two_pop_velocity(t,u_in/x,x,sig_g,v_gas,T,alpha,m_star,a_0,V_FRAG,RHO_S,E_drift,nogrowth=nogrowth)
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
            if dt<1.0:
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
            nu_gas     = alpha * k_b*T/mu/m_p * sqrt(x**3/Grav/m_star)
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
            solution_g[snap_count,:] = u_gas/x
            v_bar[snap_count,:]      = v
            vgas[snap_count,:]       = v_gas
            Diff[snap_count,:]       = D
            #
            # store the rest
            #
            v_0[snap_count,:]   = res[2]
            v_1[snap_count,:]   = res[3]
            a_t[snap_count,:]   = res[4]
            a_df[snap_count,:]  = res[5]
            a_fr[snap_count,:]  = res[6]
            a_dr[snap_count,:]  = res[7]

    progress_bar(100.,'toy model running')

    return [time,solution_d,solution_g,v_bar,vgas,v_0,v_1,a_dr,a_fr,a_df,a_t]


def two_pop_velocity(t,sigma_d_t,x,sigma_g,v_gas,T,alpha,m_star,a_0,V_FRAG,RHO_S,E_drift,nogrowth=False):
    """
    This model takes a snapshot of temperature, gas surface density and so on
    and calculates values of the representative sizes and velocities which are
    used in the two population model.
    
    USAGE:
    [v_bar,D,v_0,v_1,a_max_t,a_df,a_fr,a_dr] = ...
    two_pop_velocity(t,sigma_d_t,x,sigma_g,v_gas,T,alpha,m_star,a_0,V_FRAG,RHO_S,E_drift)
    
    WHERE:
        t            = time at which to calculate the values  [s]
        sigma_d_t    = current dust surface density array (nr)[g cm^-2]
        x            = nr radial grid points (nr)             [cm]
        timesteps    = times of the snapshots (nt)            [s]
        sigma_g      = gas surface density (nr)               [g cm^-2]
        v_gas        = gas radial velocity (nr)               [cm s^-1]
        T            = temperature snapshots (nr)             [K]
        alpha        = turbulence parameter (nr)              [-]
        m_star       = stellar mass                           [g]
        a_0          = monomer size                           [cm]
        V_FRAG       = fragmentation velocity                 [cm s^-1]
        RHO_S        = dust internal density                  [g cm^-3]
        E_drift      = drift efficiency                       [-]
    
    KEYWORD:
        nogrowth     = wether a fixed particle size is used   [False]
    
    RETURNS:
        v_bar        = the mass averaged velocity (nt,nr)         [cm s^-1]
        D            = t-interpolated diffusivity (nt,nr)         [cm^2 s^-1]
        v_0          = t-interpolated vel. of small dust (nt,nr)  [cm s^-1]
        v_1          = t-interpolated vel. of large dust (nt,nr)  [cm s^-1]
        a_max_t      = maximum grain size (nt,nr)                 [cm]
        a_df         = the fragmentation-by-drift limit (nt,nr)   [cm]
        a_fr         = the fragmentation limit (nt,nr)            [cm]
        a_dr         = the drift limit (nt,nr)                    [cm]
    """
    fudge_fr = 0.37
    fudge_dr = 0.55
    #
    # set some constants
    #
    from const import k_b,mu,m_p,Grav
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
        a_dr  = fudge_dr/E_drift*2/pi*sigma_d_t/RHO_S*x**2*(Grav*m_star/x**3)/(abs(gamma)*(k_b*T/mu/m_p))
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
        tau_grow    = sigma_g/maximum(1e-100,sigma_d_t*o_k)
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
    
        Perform one time step for the following PDE:
    
           du    d  /    \    d  /              d  /       u   \ \
           -- + -- | u v | - -- | h(x) Diff(x) -- | g(x) ----  | | = K + L u
           dt   dx \    /    dx \              dx \      h(x) / /
    
        with boundary conditions
    
            dgu/h |            |
          p ----- |      + q u |       = r
             dx   |x=xbc       |x=xbc
    Arguments:
          n_x   = # of grid points
          x     = the grid
          Diff  = value of Diff @ cell center
          v     = the values for v @ interface (array[i] = value @ i-1/2)
          g     = the values for g(x)
          h     = the values for h(x)
          K     = the values for K(x)
          L     = the values for L(x)
          flim  = diffusion flux limiting factor at interfaces
          u     = the current values of u(x)
          dt    = the time step
    
    OUTPUT:
          u     = the updated values of u(x) after timestep dt
    
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
    :    lower diagonal entries
    
    b : array
    :    diagonal entries
    
    c : array
    :    upper diagonal entries
    
    r : array
    :    right hand side vector
    
    n : int
    size of the vectors
    
    Returns:
    --------
    
    u : array
    :    solution vector
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
    :    The percentage of completion, should be
         between 0 and 100. Only 100.0 finishes with the
         word "Done!".
         
    Keywords:
    ---------
    
    text : str
    :    Possible text for describing the running process.
    
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
except (ImportError,SyntaxError):
    import os      as _os
    import inspect as _inspect
    import urllib  as _urllib
    
    print('could not import distribution_reconstruction.py, will download it and retry')
    
    _module_path = _os.path.dirname(_os.path.abspath(_inspect.getfile(_inspect.currentframe())))
    for _file in _helperfiles:
        open(_module_path+_os.sep+_file, 'wb').write(_urllib.urlopen(_gitpath+_file).read())
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