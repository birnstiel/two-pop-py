# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from .const import k_b, mu, m_p, Grav, pi, sig_h2, M_sun, AU, year


def dlydlx(x, R):
    """
    calculates the log-derivative

     dlog(y)
    -------- = dlydlx(x,y)
     dlog(x)
    """
    #
    # define the interpolation function (one for each row)
    #
    r = np.zeros(np.shape(R))
    if len(np.shape(R)) > 1:
        for i, row in enumerate(R):
            def R_int(x_int):
                return 10**(np.interp(np.log10(x_int), np.log10(x), np.log10(row)))
            h = x / 100.
            r[i] = x / row * (R_int(x + h) - R_int(x - h)) / (2. * h)
    else:
        def R_int(x_int):
            return 10**(np.interp(np.log10(x_int), np.log10(x), np.log10(R)))
        h = x / 100.
        r = x / R * (R_int(x + h) - R_int(x - h)) / (2. * h)
    return r


def trace_line_though_grid(xi, yi, f, x=None, y=None):
    """
    Returns the cell indices through which the curve moves

    """
    if x is None:
        x = 0.5 * (xi[1:] + xi[:-1])
    if y is None:
        y = 0.5 * (yi[1:] + yi[:-1])

    def fill_cells(x, y, yi, ixstart, iystart, yend):
        """
        Takes an index, returns, all cells from (including) this index
        up until the last cell that includes the value yend (towards yend)

        """
        iyend = np.searchsorted(yi, yend) - 1
        direction = int(np.sign(yend - y[iystart]))
        return [(ixstart, iy) for iy in range(iystart, min(max(0, iyend), len(y) - 1) + direction, direction)]
    #
    # begin function
    #
    fx = f(x)
    result = set()
    #
    # find first cell center where the function value is on the grid
    #
    mask = np.where((fx <= yi[-1]) & (fx >= yi[0]))[0]
    if len(mask) == 0:
        return result
    ix0 = mask[0]
    y_interface = f(xi[ix0])

    if y_interface > yi[-1]:
        iy0 = len(y) - 1
        dum = fill_cells(x, y, yi, ix0, iy0, y_interface)
        result = result.union(dum)
    elif y_interface > yi[-2]:
        ix0 = max(0, ix0 - 1)
        iy0 = len(y) - 1
        dum = fill_cells(x, y, yi, ix0, iy0, y_interface)
        result = result.union(dum)
        ix0 += 1
    elif y_interface > yi[0]:
        ix0 = max(0, ix0 - 1)
        iy0 = np.searchsorted(yi, y_interface) - 1
        dum = fill_cells(x, y, yi, ix0, iy0, y_interface)
        result = result.union(dum)
        ix0 += 1
    else:
        iy0 = 0
        dum = fill_cells(x, y, yi, ix0, iy0, y_interface)
        result = result.union(dum)

    iy0 = dum[-1][-1]
    dum = fill_cells(x, y, yi, ix0, iy0, fx[ix0])
    result = result.union(dum)
    iy0 = dum[-1][-1]

    while ix0 <= mask[-1]:
        #
        # evaluate function at next interface
        #
        y_interface = f(xi[ix0 + 1])
        #
        # fill until interface value is reached
        #
        dum = fill_cells(x, y, yi, ix0, iy0, y_interface)
        result = result.union(set(dum))
        #
        # go right
        #
        ix0 += 1  # update 1
        iy0 = dum[-1][-1]
        if ix0 > mask[-1]:
            break
        #
        # fill until final value is reached
        #
        dum = fill_cells(x, y, yi, ix0, iy0, fx[ix0])
        result = result.union(set(dum))
        iy0 = dum[-1][-1]
    #
    # transform into sorted list
    #
    result = [list(i) for i in list(result)]
    result.sort()
    return result


def reconstruct_size_distribution(r, a, t, sig_g, sig_d, alpha, rho_s, T, M_star, v_f, a_0=1e-4, fix_pd=None, ir_0=2, return_a=False):
    """
    Reconstructs the approximate size distribution based on the recipe of Birnstiel et al. 2015, ApJ.

    Arguments:
    ----------

    r : array
    :    radial grid [cm]

    a : array
    :    grain size grid, should have plenty of range and bins to work [cm]

    t : float
    :    time of the snapshot [s]

    sig_g : array
    :    gas surface densities on grid r [g cm^-2]

    sig_d : array
    :    dust surface densities on grid r [g cm^-2]

    alpha : array
    :    alpha parameter on grid r [-]

    rho_s : float
    :    material (bulk) density of the dust grains [g cm^-3]

    T : array
    :    temperature on grid r [K]

    M_star : float
    :    stellar mass [g]

    v_f : float
    :    fragmentation velocity [cm s^-1]

    Keywords:
    ---------

    a_0 : float
    :    initial particle size [cm]

    fix_pd : None | float
    :    float: set the inward diffusion slope to this values
         None:  calculate it

    return_a : bool
        If True, then in addition to the default output, also the
        three size limits: drift, fragmentation, growth timescale

    Output:
    -------
    sig_sol,a_max,r_f,sig_1,sig_2,sig_3,[a_dr,a_fr,a_grow]

    sig_sol : array
    :    2D grain size distribution on grid r x a

    a_max : array
    :    maximum particle size on grid r [cm]

    r_f : float
    :    critical fragmentation radius (see paper) [cm]

    sig_1, sig_2, sig_3 : array
    :    grain size distributions corresponding to the regions discussed in the paper
    """
    floor = 1e-100
    if fix_pd is not None:
        print('WARNING: fixing the inward diffusion slope')
    #
    # calculate derived quantities
    #
    alpha = alpha * np.ones(len(r))
    cs = np.sqrt(k_b * T / mu / m_p)
    om = np.sqrt(Grav * M_star / r**3)
    vk = r * om
    gamma = dlydlx(r, sig_g * np.sqrt(T) * om)
    p = -dlydlx(r, sig_g)
    q = -dlydlx(r, T)
    #
    # fragmentation size
    #
    b = 3. * alpha * cs**2 / v_f**2
    a_fr = sig_g / (pi * rho_s) * (b - np.sqrt(b**2 - 4.))
    a_fr[np.isnan(a_fr)] = np.inf
    #
    # drift size
    #
    a_dr = 0.55 * 2 / pi * sig_d / rho_s * r**2. * \
        (Grav * M_star / r**3) / (abs(gamma) * cs**2)
    #
    # assume erosion or something else (boundcing?) limits particles to
    # St = 1
    # and we treat it like fragmentation
    #
    a_St1 = 2.0 * sig_g / (pi * rho_s)
    a_fr = np.minimum(a_St1, a_fr)
    a_dr = np.minimum(a_St1, a_dr)
    #
    # time dependent growth
    #
    t_grow = sig_g / (om * sig_d)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'overflow encountered in exp')
        a_grow = a_0 * np.exp(t / t_grow)
    #
    # the minimum of all of those
    #
    a_max = np.minimum(np.minimum(a_fr, a_dr), a_grow)
    if a_max.max() > a[-1]:
        raise ValueError(
            'Maximum grain size larger than size grid. Increase upper end of size grid.')
    #
    # transition to turblent velocities
    #
    Re = alpha * sig_g * sig_h2 / (2 * mu * m_p)
    a_bt = (8 * sig_g / (pi * rho_s) * Re**-0.25 * np.sqrt(mu *
                                                           m_p / (3 * pi * alpha)) * (4 * pi / 3 * rho_s)**-0.5)**0.4
    #
    # apply the reconstruction recipes
    #
    n_r = len(r)
    n_a = len(a)
    sig_1 = floor * np.ones([len(a), n_r])  # the fragment distribution
    sig_2 = floor * np.ones(sig_1.shape)  # outward mixed fragments
    sig_3 = floor * np.ones(sig_1.shape)  # drift distribution diffused
    prev_frag = None
    #
    # Radial loop: to fill in all fragmentation regions
    #
    frag_mask = a_max == a_fr
    frag_idx = np.where(frag_mask)[0]
    #
    # select which cells to avoid
    #
    if ir_0 is None:
        ir_0 = np.where(a_max[:-1] > a_max[1:])[0]
        if len(ir_0) == 0:
            ir_0 = 0
        else:
            ir_0 = min(10, ir_0[0])
    #
    # calculate fragmentation effects
    #
    for ir in range(ir_0, n_r):
        ia_max = abs(a - a_max[ir]).argmin()
        if frag_mask[ir]:
            #
            # fragmentation case
            #
            i_bt = abs(a - a_bt[ir]).argmin()
            dist = (a / a[0])**1.5
            # can be 1/4, or 1/2, or average 0.375, OO13 used 1/4
            dist[i_bt:] = dist[i_bt] * (a[i_bt:] / a[i_bt])**0.375
            dist[ia_max + 1:] = floor
            dist = dist / sum(dist) * sig_d[ir]
            sig_1[:, ir] = dist
            prev_frag = ir
        elif sig_g[ir] > 1e-5:
            #
            # outward diffusion of fragments from index prev_frag
            #
            if prev_frag is not None:
                sig_2[:, ir] = sig_1[:, prev_frag] * sig_g[ir] / \
                    sig_g[prev_frag] * (r[ir] / r[prev_frag])**-1.5
                #
                # limit outward diffusion by drift
                #
                St = a * rho_s / sig_g[ir] * pi / 2.
                vd = 1. / (St + 1. / St) * cs[ir]**2 / vk[ir] * gamma[ir]
                t_dri = r[ir] / abs(vd)
                t_dif = (r[ir] - r[prev_frag])**2 / \
                    (alpha[ir] * cs[ir]**2 / om[ir] / (1. + St**2))
                #
                # the factor of 3 below changes how far out we take the
                # diffusion, but doesn't affect the results too much
                #
                sig_2[t_dif > 3 * t_dri, ir] = floor
                #
                # we also need to make sure we don't diffuse over
                # the drift limit (too much, at least)
                #
                sig_2[a > a_max[ir], ir] = floor
    #
    # ---------------------
    # add up all the the radial approximations from all cells crossed by the drift limit
    # ---------------------
    #
    # find all intersected grid cells
    # we will assume that the interfaces are in the middle of the grid center
    # usually it's the other way around, but this doesn't really matter here
    #
    ri = 0.5 * (r[1:] + r[:-1])
    ai = 0.5 * (a[1:] + a[:-1])
    ri = np.hstack((r[0] - (ri[0] - r[0]), ri, r[-1] + (r[-1] - ri[-1])))
    ai = np.hstack((a[0] - (ai[0] - a[0]), ai, a[-1] + (a[-1] - ai[-1])))
    f = interp1d(np.log10(np.hstack((0.5 * ri[0], r))), np.log10(np.hstack(
        (a_max[0], a_max))), bounds_error=False, fill_value=np.log10(a_max[-1]))
    res = trace_line_though_grid(np.log10(ri), np.log10(ai), f)
    #
    # loop through every cell
    #
    for cell in res:
        ir, ia = cell
        #
        # if fragmentation limited: skip this cell
        #
        if frag_mask[ir]:
            continue
        #
        # skip inner possibly problematic cells
        #
        if ir <= ir_0:
            continue
        #
        # find ra, our starting point, and dust and gas densities there
        #
        mask = np.arange(max(ir - 2, 0), min(ir + 2, len(r) - 1) + 1)
        # needs to be sorted to work for interpolation?
        mask = mask[a_max[mask].argsort()]
        ra = 10.**np.interp(np.log10(a[ia]),
                            np.log10(a_max[mask]), np.log10(r[mask]))
        _sigd = 10.**np.interp(np.log10(ra), np.log10(r),
                               np.log10(sig_d + 1e-100))
        _sigg = 10.**np.interp(np.log10(ra), np.log10(r),
                               np.log10(sig_g + 1e-100))
        #
        # find previous and next fragmentation index
        #
        prev_frag = frag_idx[frag_idx < ir]
        if len(prev_frag) == 0:
            prev_frag = ir_0
        else:
            prev_frag = prev_frag[-1]
        next_frag = frag_idx[frag_idx > ir]
        if len(next_frag) == 0:
            next_frag = n_r - 1
        else:
            next_frag = next_frag[0]
        #
        # get semi-analytical estimate of the dust power-law (assuming everything is a power-law)
        # v is approximated as v0 * (r/ra)**d
        #
        # old way: d = (p-q+0.5)
        #
        mask = np.zeros(n_r, dtype=bool)
        St = a[ia] * rho_s * pi / (2 * sig_g)
        v0 = -1. / (St + 1. / St) * cs**2 / vk * (p + (q + 3.) / 2.)
        d = dlydlx(r, v0)
        d[np.isnan(d)] = (p - q + 0.5)[np.isnan(d)]
        v = v0[ir] * (r / ra)**d[ir]
        pd_est = 1. / (2 * alpha) * (v / cs * vk / cs - 2 * p * alpha + vk / cs * np.sqrt(
            (v / cs)**2 + 4 * (1 + d - p) * v / vk * alpha + 4 * alpha * sig_d / sig_g))
        if fix_pd is not None:
            pd_est = fix_pd * np.ones(n_r)
        #
        # now decide where to apply the solution: inward or outward
        #
        if v0[ir] <= 0:
            # inward drift
            mask[prev_frag + 1:min(ir + 1, n_r)] = True
        else:
            # outward drift
            mask[ir + 1:next_frag + 1] = True
        #
        # mask away NaNs
        #
        mask &= np.invert(np.isnan(pd_est))
        if len(pd_est[mask]) != 0:
            inte = cumtrapz(pd_est[mask], x=np.log10(r[mask]), initial=0)
            inte /= np.abs(inte).max()
            sol = np.exp(inte)
            sol = sol / np.interp(ra, r[mask], sol) * _sigd
            sig_3[ia, mask] = np.maximum(sol, sig_3[ia, mask])
        else:
            #
            # in case there is no place to apply a solution
            # at least fill the initial cell
            #
            sig_3[ia, ir] = np.maximum(sig_d[ir], sig_3[ia, ir])
        if v0[ir] <= 0:
            #
            # outward diffusion
            #
            mask = np.zeros(n_r, dtype=bool)
            mask[ir + 1:next_frag + 1] = True
            A = a[ia] * rho_s * pi * gamma[ir] / \
                (2 * alpha[ir] * _sigg * p[ir])
            sol2 = np.maximum(sig_3[ia, mask], _sigd *
                              np.exp(A * ((r[mask] / ra)**p[mask] - 1.)))
            #
            # make sure this diffusion is not increasing
            #
            mask_idx = np.where(sol2[:-1] < sol2[1:])[0]
            if len(mask_idx) == 0:
                mask_idx = 0
            else:
                mask_idx = mask_idx[0]
            mask &= (np.arange(n_r) <= np.arange(n_r)[mask][mask_idx])
            sig_3[ia, mask] = sol2[:mask_idx + 1]

    sig_3[np.isnan(sig_3)] = floor
    #
    # extrapolate empty small-grain bins in the drift limit
    #
    for ir in range(n_r):
        if a_max[ir] == a_fr[ir]:
            continue
        #
        # if all bins are empty, fill in smallest sizes
        #
        if sum(sig_3[:, ir]) <= 10 * n_a * floor:
            mask = a <= a_0
            sig_3[mask, ir] = a[mask]**0.5
            sig_3[mask, ir] = sig_3[mask, ir] / \
                sum(sig_3[mask, ir]) * sig_d[ir]
            sig_3[np.invert(mask), ir] = floor / (1e2 * n_a)
            continue
        #
        # if there are some empty bins at the bottom, find largest empty bin
        #
        if sig_3[0, ir] <= 1e-70:
            i_full = np.where(sig_3[:, ir] > 1e-70)[0]
            if len(i_full) == 0:
                continue
            else:
                i_full = i_full[0]
        else:
            continue
        #
        # now get the slope and continue it downward
        #
        imax = sig_3[:, ir].argmax()
        pwl = np.log(sig_3[imax, ir] / sig_3[i_full, ir]) / \
            np.log(a[imax] / a[i_full])
        if imax == i_full:
            imax = np.where(sig_3[:, ir] > floor)[0][-1]
            if imax == i_full:
                # if it is still the same, we will just take a steep power-law
                pwl = -3
            else:
                pwl = np.log(sig_3[imax, ir] / sig_3[i_full, ir]
                             ) / np.log(a[imax] / a[i_full])
        sig_3[:i_full, ir] = sig_3[i_full, ir] * (a[:i_full] / a[i_full])**pwl
    #
    # Now the problem is how to stitch the 3 distributions together,
    # particularly sig_2 and sig_3. sig_2 near the fragmentation zone
    # is pretty much normalized, but further away, the drift zone might
    # dominate. So for now, we assume sig_2 to be normalized and we
    # adapt the normalization of sig_3 to account for that. It can happen
    # that the sufrace density in sig_2 is actually larger than the total
    # dust surface density. Fort this reason, we make sure sig_3 is not
    # normalized to a negative value and we renormalize the total
    # distribution then once more.
    #
    sig_3_sum = sig_3.sum(0)
    mask = sig_3_sum > 1e-50
    sig_3[:, mask] = sig_3[:, mask] / sig_3_sum[mask] * \
        np.maximum(1e-100, sig_d[mask] - sig_2.sum(0)[mask])
    #
    # add up all of them and normalize
    #
    sig_dr = sig_1 + sig_2 + sig_3
    sig_dr = sig_dr / sig_dr.sum(0) * sig_d
    #
    # as some aspects of diffusion are not included it might be useful to do a smoothing of the resulting
    # distribution.
    #
    sig_s = np.zeros(sig_dr.shape)
    for ir in np.arange(n_r):
        St = a * rho_s / sig_g[ir] * pi / 2.
        vd = 1. / (St + 1. / St) * cs[ir]**2 / vk[ir] * gamma[ir]
        #
        # radial width of the kernel: how far can particles diffuse in X orbits
        #
        X = 20.
        sig_r = np.sqrt(
            2 * pi * X / om[ir] * (alpha[ir] * cs[ir]**2 / om[ir] / (1. + St**2)))
        #
        # radial width of the kernel: how far can it diffuse in X drift time scales
        #
        # X     = 1.
        # sig_r = np.sqrt(  X*r[ir]/abs(vd) * (alpha[ir]*cs[ir]**2/om[ir]/(1.+St**2))  )
        #
        # now define a radial stencil over which to smooth (2*sigma left and 2*sigma
        # right), but at at least 1 grid left and 1 grid right, apart from the boundaries
        #
        dr_max = 2 * sig_r.max()
        ir0 = max(0, min(ir - 1, np.searchsorted(r, r[ir] - dr_max)))
        ir1 = min(n_r - 1, max(ir + 1, np.searchsorted(r, r[ir] + dr_max)))
        for ia in np.arange(n_a):
            #
            # define the size range, take a factor of 2 in mass, and go
            # from half-size to double-size
            #
            dmf = 2.  # log-normal width factor (from m/dmf to m*dmf)
            ia0 = max(0, min(ia - 1, np.searchsorted(a, a[ia] / 2)))
            ia1 = min(n_a - 1, max(ia + 1, np.searchsorted(a, a[ia] * 2)))
            #
            # construct the kernel
            #
            kernel_a = np.exp(-np.log(a[ia0:ia1 + 1] / a[ia])**2 / (2 * np.log(dmf**0.33)**2))
            kernel_r = np.exp(-(r[ir0:ir1 + 1] - r[ir])**2 / (2 * sig_r[ia]**2))
            kernel = np.outer(kernel_a, kernel_r)
            kernel = kernel / \
                np.trapz(2 * pi * r[ir0:ir1 + 1] *
                         kernel.sum(0), x=r[ir0:ir1 + 1])
            #
            # apply the smoothing
            #
            sig_s[ia, ir] = np.trapz(
                2 * pi * r[ir0:ir1 + 1] * (kernel * sig_dr[ia0:ia1 + 1, ir0:ir1 + 1]).sum(0), x=r[ir0:ir1 + 1])
    if len(frag_idx) == 0:
        frag_idx = [0]
    if return_a:
        return sig_s, a_max, r[frag_idx[-1]], sig_1, sig_2, sig_3, a_dr, a_fr, a_grow
    else:
        return sig_s, a_max, r[frag_idx[-1]], sig_1, sig_2, sig_3


def test_reconstruction():
    """Simiple test case for running the reconstruction"""
    #
    # ================
    # set up the model
    # ================
    #
    nr = 100
    na = 200
    ri = np.logspace(-1, 3, nr + 1) * AU
    ai = np.logspace(-4, 2, na + 1)

    r = 0.5 * (ri[1:] + ri[:-1])
    a = 0.5 * (ai[1:] + ai[:-1])

    M_star = M_sun
    M_disk = 0.05 * M_star
    rc = 60 * AU
    eps = 0.01
    rho_s = 1.2
    v_f = 1e3
    alpha = 1e-3

    sig_g = r**-1 * np.exp(-r / rc)
    sig_g = M_disk * sig_g / np.trapz(2 * pi * r * sig_g, x=r)
    sig_d = sig_g * eps
    T = 200 * (r / AU)**-0.5
    #
    # call the reconstruction routine
    #
    sig_dr, a_max, _, _, _, _ = reconstruct_size_distribution(r, a, 1e6 * year, sig_g, sig_d, alpha, rho_s, T, M_star, v_f)
    #
    # ========
    # PLOTTING
    # ========
    #
    _, ax = plt.subplots()
    #
    # plot the grid
    #
    for _ai in ai:
        ax.plot(ri / AU, _ai * np.ones(len(ri)), 'k')
    for _ri in ri:
        ax.plot(_ri / AU * np.ones(len(ai)), ai, 'k')
    for _a in a:
        ax.plot(r / AU, _a * np.ones(len(r)), 'kx')
    #
    # find all intersected grid cells
    #
    f = interp1d(np.log10(r), np.log10(a_max), bounds_error=False,
                 fill_value=np.log10(a_max[-1]))
    res = trace_line_though_grid(np.log10(ri), np.log10(ai), f)
    #
    # draw all intersected cells
    #
    for j, i in res:
        ax.plot(np.array([ri[j], ri[j + 1], ri[j + 1], ri[j], ri[j]]
                         ) / AU, [ai[i], ai[i], ai[i + 1], ai[i + 1], ai[i]], 'r')
    #
    # plot the distribution
    #
    mx = np.ceil(np.log10(sig_dr.max()))
    ax.contourf(r / AU, a, np.log10(sig_dr),
                np.arange(mx - 10, mx + 1), alpha=0.5)
    #
    # plot maximum particle size
    #
    ax.plot(r / AU, a_max, 'r-+')

    ax.set_xlim(0.9 * ri[0] / AU, 1.1 * ri[-1] / AU)
    ax.set_ylim(0.9 * ai[0], 1.1 * ai[-1])

    ax.set_xscale('log')
    ax.set_yscale('log')
