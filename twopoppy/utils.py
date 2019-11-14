from .const import k_b, mu, m_p, Grav, sig_h2
import numpy as np


def get_size_limits(t, sigma_d_t, x, sigma_g, v_gas, T, alpha, m_star, a_0, V_FRAG, RHO_S, E_drift, stokesregime=False, E_stick=1., nogrowth=False):
    """
    This model takes a snapshot of temperature, gas surface density and so on
    and calculates the representative sizes which are
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

    a_max : array
        maximum grain size (nr)                 [cm]

    a_df : array
        the fragmentation-by-drift limit (nr)   [cm]

    a_fr : array
        the fragmentation limit (nr)            [cm]

    a_dr : array
        the drift limit (nr)                    [cm]

    St_0: array
        St_0 from two-population approach

    St_1: array
        St_1 from two-population approach

    mask_drift : array
        boolean array where drift limit applies

    gamma : array
        pressure exponent

    o_k : array
        keplerian frequency

    """
    fudge_fr = 0.75
    fudge_dr = 0.97

    n_r = len(x)
    #
    # calculate the pressure power-law index
    #
    P = sigma_g * np.sqrt(Grav * m_star / x**3) * np.sqrt(k_b * T / mu / m_p)
    gamma = np.zeros(n_r)
    gamma[1:n_r - 1] = x[1:n_r - 1] / P[1:n_r - 1] * \
        (P[2:n_r] - P[0:n_r - 2]) / (x[2:n_r] - x[0:n_r - 2])
    gamma[0] = gamma[1]
    gamma[-1] = gamma[-2]

    #
    # calculate the sizes
    #
    o_k = np.sqrt(Grav * m_star / x**3)
    #
    # calculate the mean free path of the particles
    #
    cs = np.sqrt(k_b * T / mu / m_p)
    H = cs / o_k
    n = sigma_g / (np.sqrt(2.0 * np.pi) * H * mu * m_p)
    lambd = 0.5 / (sig_h2 * n)
    if nogrowth:
        mask = np.ones(n_r) == 1  # noqa
        a_max = a_0 * np.ones(n_r)  # noqa
        a_max_t = a_max         # noqa
        a_max_t_out = a_max         # noqa
        a_fr = a_max         # noqa
        a_dr = a_max         # noqa
        a_df = a_max         # noqa
        mask_drift = np.zeros_like(a_df, dtype=bool)
    else:
        a_fr_ep = fudge_fr * 2 * sigma_g * V_FRAG**2 / \
            (3 * np.pi * alpha * RHO_S * k_b * T / mu / m_p)
        # calculate the grain size in case of the Stokes regime
        if stokesregime:
            a_fr_stokes = np.sqrt(3 / (2 * np.pi)) * np.sqrt((sigma_g * lambd) / (alpha * RHO_S)) * V_FRAG / (np.sqrt(k_b * T / mu / m_p))
            a_fr = np.minimum(a_fr_ep, a_fr_stokes)
        else:
            a_fr = a_fr_ep
        a_dr = E_stick * fudge_dr / E_drift * 2 / np.pi * sigma_d_t / RHO_S * \
            x**2 * (Grav * m_star / x**3) / (abs(gamma) * (k_b * T / mu / m_p))
        N = 0.5
        a_df = fudge_fr * 2 * sigma_g / (RHO_S * np.pi) * V_FRAG * np.sqrt(
            Grav * m_star / x) / (abs(gamma) * k_b * T / mu / m_p * (1 - N))
        a_max = np.maximum(a_0 * np.ones(n_r), np.minimum(a_dr, a_fr))

        ###
        # EXPERIMENTAL: inlcude a_df as upper limit
        a_max = np.maximum(a_0 * np.ones(n_r), np.minimum(a_df, a_max))
        a_max_out = np.minimum(a_df, a_max)
        # mask      = all([a_dr<a_fr,a_dr<a_df],0)
        mask_drift = np.array([adr < afr and adr < adf for adr, afr, adf in zip(a_dr, a_fr, a_df)])

        ###
        #
        # calculate the growth time scale and thus a_1(t)
        #
        tau_grow = sigma_g / np.maximum(1e-100, E_stick * sigma_d_t * o_k)
        a_grow = a_0 * np.exp(np.minimum(709.0, t / tau_grow))
        a_max_t = np.minimum(a_max, a_grow)
        a_max_t_out = np.minimum(a_max_out, a_grow)

    #
    # calculate the Stokes number of the particles
    #
    St_0 = RHO_S / sigma_g * np.pi / 2 * a_0
    St_1 = RHO_S / sigma_g * np.pi / 2 * a_max_t

    return {
        'St_0': St_0,
        'St_1': St_1,
        'a_max': a_max_t_out,
        'a_df': a_df,
        'a_fr': a_fr,
        'a_dr': a_dr,
        'a_grow': a_grow,
        'mask_drift': mask_drift,
        'gamma': gamma,
        'o_k': o_k,
        }


def get_velocities_diffusion(x, gamma, v_gas, St_0, St_1, T, o_k, alpha, mask_drift):
    """Calculate the velocities and diffusion constants in the two-pop approach.

    Parameters
    ----------
    x : array
        radial graid
    gamma : array
        pressure exponent
    v_gas : array
        gas velocity
    St_0 : array
        St_0 from the two-population paper
    St_1 : array
        St_1 from the two-population paper
    T : array
        temperature array
    o_k : array
        keplerian frequency
    alpha : array
        turbulence parameter
    mask_drift : array
        boolean mask where the drift limit applies

    Returns
    -------
    dict:
        'v_bar': averaged two-pop velocity
        'D': diffusion constant
        'v_0': small grain velocity
        'v_1': large grain velocity
        'f_m': f_m from two-pop. paper

    """
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
    f_m = 0.75 * np.invert(mask_drift) + 0.97 * mask_drift
    #
    # calculate the mass weighted transport velocity
    #
    v_bar = v_0 * (1 - f_m) + v_1 * f_m
    #
    # calculate the diffusivity
    #
    D = alpha * k_b * T / mu / m_p / o_k

    return {
        'v_bar': v_bar,
        'D': D,
        'v_0': v_0,
        'v_1': v_1,
        'f_m': f_m}
