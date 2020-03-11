def run(x, a_0, time, sig_g, sig_d, v_gas, T, alpha, m_star, V_FRAG, RHO_S, E_drift, E_stick=1., stokesregime=False, nogrowth=False, gasevol=True, alpha_gas=None):

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

    alpha_gas : None | array | function
        if not None: use this for the gas [-]


    Keywords:
    ---------

    stokesregime : bool
        true: also treat the stokes drag regime [False]

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
    from numpy import ones, zeros, maximum, minimum, sqrt, where
    from .const import year, Grav, k_b, mu, m_p
    from .utils import get_size_limits, get_velocities_diffusion
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
    a_gr            = zeros([n_t,n_r])  # noqa
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
    size_limits = get_size_limits(t, solution_d[0, :], x, sig_g, v_gas, Tfunc(x, locals()),
                                  alpha_func(x, locals()), m_star, a_0, V_FRAG, RHO_S,
                                  E_drift, E_stick=E_stick, stokesregime=stokesregime, nogrowth=nogrowth)

    gamma = size_limits['gamma']
    St_0 = size_limits['St_0']
    St_1 = size_limits['St_1']
    o_k = size_limits['o_k']
    mask_drift = size_limits['mask_drift']
    # calculate the velocity

    velocities = get_velocities_diffusion(x, gamma, v_gas, St_0, St_1, T, o_k, alpha, mask_drift)


    v_bar[0, :]       = velocities['v_bar']         # noqa
    Diff[0, :]        = velocities['D']             # noqa
    v_0[0, :]         = velocities['v_0']           # noqa
    v_1[0, :]         = velocities['v_1']           # noqa
    a_t[0, :]         = size_limits['a_max']        # noqa
    a_df[0, :]        = size_limits['a_df']         # noqa
    a_fr[0, :]        = size_limits['a_fr']         # noqa
    a_gr[0, :]        = size_limits['a_grow']       # noqa
    a_dr[0, :]        = size_limits['a_dr']         # noqa
    Tout[0, :]        = Tfunc(x, locals())          # noqa
    alphaout[0, :]    = alpha_func(x, locals())     # noqa
    alphagasout[0, :] = alpha_gas_func(x, locals()) # noqa

    #
    # the loop
    #
    dt = 10 * year
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

        # calculate the sizes

        size_limits = get_size_limits(t, u_in / x, x, sig_g, v_gas, _T, _alpha, m_star,
                                      a_0, V_FRAG, RHO_S, E_drift, E_stick=E_stick,
                                      stokesregime=stokesregime, nogrowth=nogrowth, a_grow_prev=size_limits['a_grow'])

        gamma = size_limits['gamma']
        St_0 = size_limits['St_0']
        St_1 = size_limits['St_1']
        o_k = size_limits['o_k']
        mask_drift = size_limits['mask_drift']
        # calculate the velocity

        velocities = get_velocities_diffusion(x, gamma, v_gas, St_0, St_1, T, o_k, alpha, mask_drift)

        v = velocities['v_bar']
        D = velocities['D']

        v[0]   = v[1]   # noqa
        D[0]   = D[1]   # noqa
        D[-2:] = 0      # noqa
        v[-2:] = 0      # noqa
        #
        # set up the equation
        #
        h = sig_g * x
        #
        # do the update
        #
        u_dust = impl_donorcell_adv_diff_delta(
            n_r, x, D, v, g, h, K, L, flim, u_in, dt, 1, 1, 0, 0, 0, 0, 1, A0, B0, C0, D0)

        mask = abs(u_dust[2:-1] / u_in[2:-1] - 1) > 2
        #
        # try variable time step
        #
        while any(u_dust[2:-1][mask] / x[2:-1][mask] >= 1e-30):
            dt = dt / 10.
            if dt < year and snap_count > 0:
                print('ERROR: time step got too short')
                sys.exit(1)
            u_dust = impl_donorcell_adv_diff_delta(
                n_r, x, D, v, g, h, K, L, flim, u_in, dt, 1, 1, 0, 0, 0, 0, 1, A0, B0, C0, D0)
            mask = abs(u_dust[2:-1] / u_in[2:-1] - 1) > 0.3
        #
        # update
        #
        u_in = u_dust[:]
        t = t + dt
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
            v_bar[snap_count, :]      = v           # noqa
            vgas[snap_count, :]       = v_gas       # noqa
            Diff[snap_count, :]       = D           # noqa
            #
            # store the rest
            #
            v_0[snap_count, :]      = velocities['v_0']     # noqa
            v_1[snap_count, :]      = velocities['v_1']     # noqa
            a_t[snap_count, :]      = size_limits['a_max']  # noqa
            a_df[snap_count, :]     = size_limits['a_df']   # noqa
            a_fr[snap_count, :]     = size_limits['a_fr']   # noqa
            a_dr[snap_count, :]     = size_limits['a_dr']   # noqa
            a_gr[snap_count, :]     = size_limits['a_grow']   # noqa
            Tout[snap_count, :]     = _T      # noqa
            alphaout[snap_count, :] = _alpha  # noqa
            alphagasout[snap_count, :] = _alpha_gas  # noqa

    progress_bar(100., 'toy model running')

    return time, solution_d, solution_g, v_bar, vgas, v_0, v_1, a_dr, a_fr, a_df, a_t, Tout, alphaout, alphagasout


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
