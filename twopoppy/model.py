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

def run(x, a_0, time, sig_g, sig_d, v_gas, T, alpha, m_star, V_FRAG, RHO_S,peff,st_min,st_max,tlife,fm_f,fm_d,dice,
        E_drift,stokes, E_stick=1., nogrowth=False, gasevol=True, alpha_gas=None, ptesim=True):
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

    alpha : array | function
        turbulence parameter (nr)       [-]

    m_star : float
        stellar mass (nt)               [g]

    V_FRAG : float
        fragmentation velocity          [cm s^-1]

    RHO_S : float
        internal density of the dust    [g cm^-3]
        
    peff : float
        pebble trapping efficiency      [-]

	st_min: float
        min. Stokes number for trapping [-]
        
	st_max: float
        max. Stokes number for trapping [-]
        
	tlife: float
        life time of a trap in orbits   [-]
    
    fm_f: float
        calibration constant for fragmentation limit [-]
        
    fm_d: float
        calibration constat for drift limit [-]
        
    dice: float
         ice content inside the snowline [-]   
    
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
        
    stokes : bool
        true: consinder both Epstein and Stokes regime [False]

    alpha_gas : None | array | function
        if not None: use this for the gas [-]


    Returns:
    ---------

    time : array
        snapshot time            (nt)          [s]

    solution_d : array
        dust column density     (nt,nr)       [g cm^-2]

    solution_g : array
        dust column density     (nt,nr)       [g cm^-2]
        
    solution_p : array
        planetesimal column density (nt,nr)    [g cm^-2]

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
    from numpy import ones, zeros, Inf, maximum, minimum, sqrt, where, pi, log10
    from .const import year, Grav, k_b, mu, m_p
    import sys
    #
    # some setup
    #
    n_r = len(x)
    n_t = len(time)
    g = ones(n_r)
    K = zeros(n_r)
    L = zeros(n_r)
    flim = ones(n_r)
    A0 = zeros(n_r)
    B0 = zeros(n_r)
    C0 = zeros(n_r)
    D0 = zeros(n_r)

    #
    # setup
    #
    t               = time[0]           # noqa
    solution_d      = zeros([n_t,n_r])  # noqa
    solution_d[0,:] = sig_d             # noqa
    solution_g      = zeros([n_t,n_r])  # noqa
    solution_g[0,:] = sig_g             # noqa
    solution_p      = zeros([n_t,n_r])
    dot_sigma_ptes  = zeros(len(x))
    ptesimal_tp     = zeros(len(x))
    vgas            = zeros([n_t,n_r])  # noqa
    vgas[0,:]       = v_gas             # noqa
    v_bar           = zeros([n_t,n_r])  # noqa
    Diff            = zeros([n_t,n_r])  # noqa
    v_0             = zeros([n_t,n_r])  # noqa
    v_1             = zeros([n_t,n_r])  # noqa
    a_t             = zeros([n_t,n_r])  # noqa
    a_df            = zeros([n_t,n_r])  # noqa
    a_fr            = zeros([n_t,n_r])  # noqa
    a_dr            = zeros([n_t,n_r])  # noqa
    Tout            = zeros([n_t,n_r])  # noqa
    alphaout        = zeros([n_t,n_r])  # noqa
    alphagasout     = zeros([n_t,n_r])  # noqa
    u_in            = solution_d[0,:]*x # noqa
    it_old          = 1                 # noqa
    snap_count      = 0                 # noqa

    progress_bar(round((it_old - 1) / (n_t - 1) * 100), 'toy model running')

    # T can be either an array or a specific function.
    # in either case, we define a function that returns the temperature

    if hasattr(T, '__call__'):
        Tfunc = T
    else:
        def Tfunc(x, a_tlocals_):
            return T

    # alpha can be either an array or a specific function.
    # in either case, we define a function that returns it

    if hasattr(alpha, '__call__'):
        alpha_func = alpha
    else:
        def alpha_func(x, locals_):
            return alpha

    # the same for alpha_gas

    if alpha_gas is None:
        alpha_gas = alpha

    if hasattr(alpha_gas, '__call__'):
        alpha_gas_func = alpha_gas
    else:
        def alpha_gas_func(x, locals_):
            return alpha_gas

    #
    # save the velocity which will be used
    #
    res = get_velocity(t, solution_d[0, :], x, sig_g, v_gas, Tfunc(x, locals()),
                       alpha_func(x, locals()), m_star, a_0, V_FRAG, RHO_S, fm_f, fm_d,
                       E_drift,stokes, E_stick=E_stick, nogrowth=nogrowth)

    v_bar[0, :]       = res[0]                      # noqa
    Diff[0, :]        = res[1]                      # noqa
    v_0[0, :]         = res[2]                      # noqa
    v_1[0, :]         = res[3]                      # noqa
    a_t[0, :]         = res[4]                      # noqa
    a_df[0, :]        = res[5]                      # noqa
    a_fr[0, :]        = res[6]                      # noqa
    a_dr[0, :]        = res[7]                      # noqa
    Tout[0, :]        = Tfunc(x, locals())          # noqa
    alphaout[0, :]    = alpha_func(x, locals())     # noqa
    alphagasout[0, :] = alpha_gas_func(x, locals()) # noqa

    #
    # the loop
    #
    dt = Inf
    while t < time[-1]:
        #
        # set the time step
        #
        dt = min(dt * 10, time[it_old] - t)
        if t != 0.0:
            dt = min(dt, t / 200.0)
        if dt == 0:
            print('ERROR:')
            print('t      = %g years' % (t / year))
            print('it_old = %g' % it_old)
            print('dt = 0')
            sys.exit(1)

        # update the temperature and alpha

        _T = Tfunc(x, locals())
        _alpha = alpha_func(x, locals())
        _alpha_gas = alpha_gas_func(x, locals())

        # calculate the velocity

        res = get_velocity(t, u_in / x, x, sig_g, v_gas, _T, _alpha, m_star,
                           a_0, V_FRAG, RHO_S, fm_f, fm_d, E_drift, stokes, E_stick=E_stick, nogrowth=nogrowth)

        v      = res[0] # noqa
        D      = res[1] # noqa
        v[0]   = v[1]   # noqa
        D[0]   = D[1]   # noqa
        D[-2:] = 0      # noqa
        v[-2:] = 0      # noqa
        a_big = res[4]
        #
        # set up the equation
        
        #
        h = sig_g * x
        #
		#
		#sig_d = u_dust/x
		# 
        # try new implicit planetesimal formation by Oliver
        #
        if ptesim:
            # size and mass of one planetesimal
            psize         = 50 * 1000. * 100. #! 50 km in cm
            ptesimal_mass = RHO_S * 4/3 * pi * psize**3 # PLANETESIMAL MASS for spherical objects with 100 km diameter
            fluxcrit_tp      = zeros(len(x)-1)
            L_0_tp = zeros(len(x)-1)
            L_1_tp = zeros(len(x)-1)
            cs    = sqrt(k_b*T[1:]/(mu*m_p)) # sound speed
            hdisk = cs * sqrt(x[1:]**3./(Grav * m_star)) # gas scale height
            dpre = 5
            #
            # calculate Stokes number in Epstein regime
            #
            #St_0 = RHO_S / sig_g[1:] * pi / 2.0 * a_0 # small pop
            #St_1 = RHO_S / sig_g[1:] * pi / 2.0 * a_t[0,1:] # large pop
            #
            # calculate Stokes numbers in both regimes
            #
            St_0 = get_St(a_0,T[1:], sig_g[1:],x[1:],m_star,RHO_S,stokes)
            St_1 = get_St(a_big[1:],T[1:], sig_g[1:],x[1:],m_star,RHO_S,stokes)
            #print('Well, well, well...')
            #
            # calculate velocity of small and large population
            #
            #
            #import pdb; pdb.set_trace()
            v_small = abs(1.0/(St_0 + 1./St_0) * \
                            minimum(cs, \
                                    1./sqrt(Grav*m_star) * k_b/(mu*m_p) * sqrt(T[1:]) * x[1:]**3 / sig_g[1:] * \
                                    1./(x[1:]-x[:-1]) * ( sig_g[1:] * sqrt(T[1:]/x[1:]**3) - sig_g[:-1] * sqrt(T[:-1]/x[:-1]**3) ) \
                                    ) \
                          )
            v_big   = abs(1.0/(St_1 + 1./St_1) * \
                            minimum(cs, \
                                    1./sqrt(Grav*m_star) * k_b/(mu*m_p) * sqrt(T[1:]) * x[1:]**3 / sig_g[1:] * \
                                    1./(x[1:]-x[:-1]) * ( sig_g[1:] * sqrt(T[1:]/x[1:]**3) - sig_g[:-1] * sqrt(T[:-1]/x[:-1]**3) ) \
                                    ) \
                          )
            mass_tr = zeros(len(x)-1)
            mass_tr_small = zeros(len(x)-1)
            mass_tr_big = zeros(len(x)-1)
            # gas scale height
            hdisk = cs * sqrt(x[1:]**3./(Grav * m_star))
            # orbit time
            torbit = 2. * pi / (sqrt((Grav*m_star)/x[1:]**3.))
            # evaporation inside the snowline
            mask_ice    = T[1:]>170 # snowline position 170 K
            dustice           = ones(len(x)-1)
            dustice[mask_ice] = dice # new value for dustice from Lodders 2003 (Table 11), formerly 1/3
            mask_small = ones(len(x)-1, dtype=bool)
            mask_big   = mask_small    			
            for i in range(1,len(x)-1):
                mask_small[i] = st_max > St_0[i] > st_min
                mask_big[i]   = st_max > St_1[i] > st_min
            #m_trapped       = zeros(len(x)-1)
            #m_trapped_small = zeros(len(x)-1)
            #m_trapped_big   = zeros(len(x)-1)
            #
            # get the trapped mass
            #
            #
            mass_tr_small[mask_small] = mass_tr_small[mask_small] + dustice[mask_small] * peff * sig_d[1:][mask_small] * v_small[mask_small] * tlife * torbit[mask_small] * 2.0 * pi * x[1:][mask_small]
            mass_tr_big[mask_big] = mass_tr_big[mask_big] + dustice[mask_big] * peff *sig_d[1:][mask_big] * v_big[mask_big] * tlife * torbit[mask_big] * 2.0 * pi * x[1:][mask_big]
            mass_tr= mass_tr_small + mass_tr_big
            # check wether trapped mass bigger than one planetesimal	
            #import pdb; pdb.set_trace()
            mask_tr = mass_tr > ptesimal_mass 
            fluxcrit_tp[mask_tr] = 1 
            mask_flux = fluxcrit_tp == 1 
            #
            # mask where trapped mass is enough to trigger planetesimal formation AND the Stokes number of small particles fits
            mask_red0 = zeros(len(mask_tr), dtype=bool)
            # mask where trapped mass is enough to trigger planetesimal formation AND the Stokes number of big particles fits
            mask_red1 = mask_red0
            for i in range(1,len(mask_tr)):
                if mask_tr[i]==mask_small[i]:
                    mask_red0[i] = mask_tr[i]
                if mask_tr[i]==mask_big[i]:
                    mask_red1[i] = mask_tr[i]
            #
            L_0_tp[mask_red0] = -peff * sqrt(Grav*m_star)/(dpre*cs[mask_red0]*sqrt(x[1:][mask_red0])**3)*v_small[mask_red0]
            L_1_tp[mask_red1] = -peff * sqrt(Grav*m_star)/(dpre*cs[mask_red1]*sqrt(x[1:][mask_red1])**3)*v_big[mask_red1]
            # need to calculate f_m properly but just for now
            #f_m_tp = 0.8
            # or use the calculated one
            f_m_tp = res[8]
            L = L_0_tp * (1-f_m_tp[1:]) + L_1_tp * f_m_tp[1:]
            # 
            # end implicit planetesimal formation
            #
        #
        # update
        #
        # do the update
        #
        u_dust = impl_donorcell_adv_diff_delta(
            n_r, x, D, v, g, h, K, L, flim, u_in, dt, 1, 1, 0, 0, 0, 0, 1, A0, B0, C0, D0)

        mask = abs(u_dust[2:-1] / u_in[2:-1] - 1) > 0.05
        #
        # try variable time step
        #
        while any(u_dust[2:-1][mask] / x[2:-1][mask] >= 1e-30):
            dt = dt / 10.
            if dt < 1.0 and snap_count > 0:
                print('ERROR: time step got too short')
                sys.exit(1)
            u_dust = impl_donorcell_adv_diff_delta(
                n_r, x, D, v, g, h, K, L, flim, u_in, dt, 1, 1, 0, 0, 0, 0, 1, A0, B0, C0, D0)
            mask = abs(u_dust[2:-1] / u_in[2:-1] - 1) > 0.3
        u_in = u_dust[:]
        t = t + dt
        sig_d = u_dust / x
		#
		# sum up the planetesimals and find the total dust mass
		#
        if ptesim:
            x05 = zeros(len(x))
            x05_distance = zeros(len(x))
            x053_distance = zeros(len(x))
            for i in range(1,len(x)):
                x05[i] = 10**(0.5*(log10(x[i-1])+log10(x[i])))
                x05_distance[i] = x05[i] - x05[i-1]
                x053_distance[i] = x05[i]**3 - x05[i-1]**3

            
            tdmass = 0
            tpmass = 0
            tpn    = 0 
            #
            #
            #
            dot_sigma_ptes[1:] = - L * sig_d[1:]
            # sum up the planetesimal column density and take the snowline into account     
            ptesimal_tp[2:] = ptesimal_tp[2:] + dot_sigma_ptes[2:] * dustice[1:] * dt
            # total dust mass after potential planetesimal formation
            #tdmass = sum(sig_d * 2. * pi * x * (x[1:]-x[:-1])*0.5)
            # total planetesimal mass
            tpmass = sum(ptesimal_tp[1:] *2.0 * pi * x[1:] * (x[1:]-x[:-1])*0.5)
            # total number of planetesimals
            tpn = tpmass / ptesimal_mass
            # ignore the first two and the last grid point due to boundary problems
            #ptesimal_tp[2] = ptesimal_tp[3]
            #ptesimal_tp[1] = ptesimal_tp[2]
            #ptesimal_tp[-1] = ptesimal_tp[-2]
		#
        # update the gas
        #
        if gasevol:
            nu_gas = _alpha_gas * k_b * _T / mu / m_p * sqrt(x**3 / Grav / m_star)
            u_gas_old = sig_g * x
            u_gas = u_gas_old[:]
            v_gas = zeros(n_r)
            D_gas = 3.0 * sqrt(x)
            g_gas = nu_gas / sqrt(x)
            h_gas = ones(n_r)
            K_gas = zeros(n_r)
            L_gas = zeros(n_r)
            p_L = -(x[1] - x[0]) * h_gas[1] / (x[1] * g_gas[1])
            q_L = 1. / x[0] - 1. / x[1] * g_gas[0] / g_gas[1] * h_gas[1] / h_gas[0]
            r_L = 0.0
            u_gas = impl_donorcell_adv_diff_delta(n_r, x, D_gas, v_gas, g_gas, h_gas, K_gas, L_gas,
                                                  flim, u_gas, dt, p_L, 0.0, q_L, 1.0, r_L, 1e-100 * x[n_r - 1], 1, A0, B0, C0, D0)
            sig_g = u_gas / x
            sig_g = maximum(sig_g, 1e-100)

            #
            # now get the gas velocities from the exact fluxes
            #
            u_flux = zeros(n_r)
            u_flux[1:n_r + 1] = - flim[1:n_r + 1] * 0.25 * (D_gas[1:] + D_gas[:-1]) * (h_gas[1:] + h_gas[:-1]) * (
                g_gas[1:] / h_gas[1] * u_gas[1:] - g_gas[:-1] / h_gas[:-1] * u_gas[:-1]) / (x[1:] - x[:-1])
            mask = u_flux > 0.0
            imask = u_flux <= 0.0
            v_gas[mask] = u_flux[mask] / u_gas[maximum(0, where(mask)[0] - 1)]
            v_gas[imask] = u_flux[imask] / u_gas[minimum(n_r - 1, where(imask)[0] + 1)]

        #
        # find out if we reached a snapshot
        #
        if t >= time[it_old]:
            #
            # one more step completed
            #
            it_old = it_old + 1
            snap_count = snap_count + 1
            #
            # notify
            #
            progress_bar(round((it_old - 1.) / (n_t) * 100.), 'toy model running')
            #
            # save the data
            #
            solution_d[snap_count, :] = u_dust / x  # noqa
            solution_g[snap_count, :] = sig_g       # noqa
            solution_p[snap_count, :] = ptesimal_tp
            v_bar[snap_count, :]      = v           # noqa
            vgas[snap_count, :]       = v_gas       # noqa
            Diff[snap_count, :]       = D           # noqa
            #
            # store the rest
            #
            v_0[snap_count, :]         = res[2]      # noqa
            v_1[snap_count, :]         = res[3]      # noqa
            a_t[snap_count, :]         = res[4]      # noqa
            a_df[snap_count, :]        = res[5]      # noqa
            a_fr[snap_count, :]        = res[6]      # noqa
            a_dr[snap_count, :]        = res[7]      # noqa
            Tout[snap_count, :]        = _T          # noqa
            alphaout[snap_count, :]    = _alpha      # noqa
            alphagasout[snap_count, :] = _alpha_gas  # noqa

    progress_bar(100., 'toy model running')

    return time, solution_d, solution_g, solution_p, v_bar, vgas, v_0, v_1, a_dr, a_fr, a_df, a_t, Tout, alphaout, alphagasout


def get_velocity(t, sigma_d_t, x, sigma_g, v_gas, T, alpha, m_star, a_0, V_FRAG, RHO_S, fm_f, fm_d, E_drift, stokesregime, E_stick=1., nogrowth=False):
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
        affects a_frag and diffusivity

    m_star : float
        stellar mass                           [g]

    a_0 : float
        monomer size                           [cm]

    V_FRAG : float
        fragmentation velocity                 [cm s^-1]

    RHO_S : float
        dust internal density                  [g cm^-3]

    fm_f: float
        calibration constant for fragmentation limit [-]
        
    fm_d: float
        calibration constat for drift limit [-]

    E_drift : float
        drift efficiency                       [-]

    Keywords:
    ---------

    E_stick : float
        sticking probability                   [-]

    nogrowth : bool
        wether a fixed particle size is used   [False]
    
    stokes : bool
        true: consinder both Epstein and Stokes regime [False]

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
    
    f_m  : mass distribution ratio (nr)         [-]
    """
    fudge_fr = fm_f
    fudge_dr = fm_d
    #
    # set some constants
    #
    from .const import k_b, mu, m_p, Grav, sig_h2
    from numpy import ones, zeros, maximum, minimum, sqrt, array, exp, invert, pi

    n_r = len(x)
    #
    # calculate the pressure power-law index
    #
    P = sigma_g * sqrt(Grav * m_star / x**3) * sqrt(k_b * T / mu / m_p)
    gamma = zeros(n_r)
    gamma[1:n_r - 1] = x[1:n_r - 1] / P[1:n_r - 1] * (P[2:n_r] - P[0:n_r - 2]) / (x[2:n_r] - x[0:n_r - 2])
    gamma[0] = gamma[1]
    gamma[-1] = gamma[-2]

    #
    # calculate the sizes
    #
    o_k = sqrt(Grav * m_star / x**3)
    #
    # calculate the mean free path of the particles
    #
    cs      = sqrt(k_b*T/mu/m_p)
    H = cs / o_k
    n   = sigma_g/(sqrt(2.0*pi)*H*mu*m_p)
    lambd   = 0.5/(sig_h2*n)
    if nogrowth:
        mask        = ones(n_r)==1  # noqa
        a_max       = a_0*ones(n_r) # noqa
        a_max_t     = a_max         # noqa
        a_max_t_out = a_max         # noqa
        a_fr        = a_max         # noqa
        a_dr        = a_max         # noqa
        a_df        = a_max         # noqa
    else:
        a_fr_ep = fudge_fr * 2 * sigma_g * V_FRAG**2 / (3 * pi * alpha * RHO_S * k_b * T / mu / m_p)
        # calculate the grain size in case of the Stokes regime
        if stokesregime:
            a_fr_stokes = sqrt(3/(2*pi)) * sqrt((sigma_g*lambd)/(alpha*RHO_S)) * V_FRAG/(sqrt(k_b*T/mu/m_p))
            a_fr = minimum(a_fr_ep, a_fr_stokes)
        else:
            a_fr = a_fr_ep
        a_dr = E_stick * fudge_dr / E_drift * 2 / pi * sigma_d_t / RHO_S * x**2 * (Grav * m_star / x**3) / (abs(gamma) * (k_b * T / mu / m_p))
        N = 0.5
        a_df = fudge_fr * 2 * sigma_g / (RHO_S * pi) * V_FRAG * sqrt(Grav * m_star / x) / (abs(gamma) * k_b * T / mu / m_p * (1 - N))
        a_max = maximum(a_0 * ones(n_r), minimum(a_dr, a_fr))

        ###
        # EXPERIMENTAL: inlcude a_df as upper limit
        a_max = maximum(a_0 * ones(n_r), minimum(a_df, a_max))
        a_max_out = minimum(a_df, a_max)
        # mask      = all([a_dr<a_fr,a_dr<a_df],0)
        mask = array([adr < afr and adr < adf for adr,
                      afr, adf in zip(a_dr, a_fr, a_df)])

        ###
        #
        # calculate the growth time scale and thus a_1(t)
        #
        tau_grow = sigma_g / maximum(1e-100, E_stick * sigma_d_t * o_k)
        a_max_t = minimum(a_max, a_0 * exp(minimum(709.0, t / tau_grow)))
        a_max_t_out = minimum(a_max_out, a_0 * exp(minimum(709.0, t / tau_grow)))

    #
    # calculate the Stokes number of the particles
    #
    St_0 = RHO_S / sigma_g * pi / 2 * a_0
    St_1 = RHO_S / sigma_g * pi / 2 * a_max_t
    #
    # calculate the velocities of the two populations:
    # First: gas velocity
    #
    v_0 = v_gas / (1 + St_0**2)
    v_1 = v_gas / (1 + St_1**2)
    #
    # Second: drift velocity
    #
    v_dr = k_b * T / mu / m_p / (2 * o_k * x) * gamma
    #
    # level of at the peak position
    #
    v_0 = v_0 + 2 / (St_0 + 1 / St_0) * v_dr
    v_1 = v_1 + 2 / (St_1 + 1 / St_1) * v_dr
    #
    # set the mass distribution ratios
    #
    f_m = 0.75 * invert(mask) + 0.97 * mask
    #
    # calculate the mass weighted transport velocity
    #
    v_bar = v_0 * (1 - f_m) + v_1 * f_m
    #
    # calculate the diffusivity
    #
    D = alpha * k_b * T / mu / m_p / o_k

    return [v_bar, D, v_0, v_1, a_max_t_out, a_df, a_fr, a_dr, f_m]


def impl_donorcell_adv_diff_delta(n_x, x, Diff, v, g, h, K, L, flim, u_in, dt, pl, pr, ql, qr, rl, rr, coagulation_method, A, B, C, D):
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
    D05 = zeros(n_x)
    h05 = zeros(n_x)
    rhs = zeros(n_x)
    #
    # calculate the arrays at the interfaces
    #
    for i in range(1, n_x):
        D05[i] = flim[i] * 0.5 * (Diff[i - 1] + Diff[i])
        h05[i] = 0.5 * (h[i - 1] + h[i])
    #
    # calculate the entries of the tridiagonal matrix
    #
    for i in range(1, n_x - 1):
        vol = 0.5 * (x[i + 1] - x[i - 1])
        A[i] = -dt / vol *  \
            (
            max(0., v[i]) +
            D05[i] * h05[i] * g[i - 1] / ((x[i] - x[i - 1]) * h[i - 1])
            )
        B[i] = 1. - dt * L[i] + dt / vol * \
            (
            max(0., v[i + 1]) -
            min(0., v[i]) +
            D05[i + 1] * h05[i + 1] * g[i] / ((x[i + 1] - x[i]) * h[i]) +
            D05[i] * h05[i] * g[i] / ((x[i] - x[i - 1]) * h[i])
            )
        C[i] = dt / vol *  \
            (
            min(0., v[i + 1]) -
            D05[i + 1] * h05[i + 1] * g[i + 1] / ((x[i + 1] - x[i]) * h[i + 1])
            )
        D[i] = -dt * K[i]
    #
    # boundary Conditions
    #
    A[0] = 0.
    B[0] = ql - pl * g[0] / (h[0] * (x[1] - x[0]))
    C[0] = pl * g[1] / (h[1] * (x[1] - x[0]))
    D[0] = u_in[0] - rl

    A[-1] = - pr * g[-2] / (h[-2] * (x[-1] - x[-2]))
    B[-1] = qr + pr * g[-1] / (h[-1] * (x[-1] - x[-2]))
    C[-1] = 0.
    D[-1] = u_in[-1] - rr

    #
    # if coagulation_method==2,
    #  then we change the arrays and
    #  give them back to the calling routine
    # otherwise, we solve the equation
    #
    if coagulation_method == 2:
        A = A / dt
        B = (B - 1.) / dt
        C = C / dt
        D = D / dt
    else:
        #
        # the old way
        #
        # rhs = u - D

        #
        # the delta-way
        #
        for i in range(1, n_x - 1):
            rhs[i] = u_in[i] - D[i] - \
                (A[i] * u_in[i - 1] + B[i] * u_in[i] + C[i] * u_in[i + 1])
        rhs[0] = rl - (B[0] * u_in[0] + C[0] * u_in[1])
        rhs[-1] = rr - (A[-1] * u_in[-2] + B[-1] * u_in[-1])

        #
        # solve for u2
        #
        u2 = tridag(A, B, C, rhs, n_x)
        #
        # update u
        # u = u2   # old way
        #
        u_out = u_in + u2  # delta way

    return u_out


def tridag(a, b, c, r, n):
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
    u = np.zeros(n)

    if b[0] == 0.:
        raise ValueError('tridag: rewrite equations')

    bet = b[0]

    u[0] = r[0] / bet

    for j in np.arange(1, n):
        gam[j] = c[j - 1] / bet
        bet = b[j] - a[j] * gam[j]

        if bet == 0:
            raise ValueError('tridag failed')
        u[j] = (r[j] - a[j] * u[j - 1]) / bet

    for j in np.arange(n - 2, -1, -1):
        u[j] = u[j] - gam[j + 1] * u[j + 1]
    return u


def progress_bar(perc, text=''):
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

    if text != '':
        text = text + ' ... '

    if perc == 100.0:
        sys.stdout.write('\r' + text + 'Done!\n')
        sys.stdout.flush()
    else:
        sys.stdout.write('\r' + text + '%d %%' % round(perc))
        sys.stdout.flush()
