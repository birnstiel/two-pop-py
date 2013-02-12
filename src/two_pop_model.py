import widget,sys
from numpy import ones,zeros,shape,arange,Inf,maximum,minimum,exp,array,sqrt,invert
from matplotlib.mlab import find
from matplotlib.pyplot import semilogx,xlim,ylim,figure
from uTILities import progress_bar,tridag

def two_pop_model_run_ic(x,a_0,time,sig_g,sig_d,v_gas,T,alpha,m_star,V_FRAG,RHO_S,peak_position,E_drift,nogrowth=False):
    """
    this function evolves the two population model (all model settings
    are stored in two_pop_velocity). It returns the important parameters of
    the model.
    
    USAGE:
    [time,solution,v_bar,v_0,v_1,a_dr,a_fr,a_df,a_t]  =  two_pop_model_run_ic ...
    (x,a_0,time,sig_g,sig_d,v_gas,T,alpha,m_star,V_FRAG,RHO_S,peak_position,E_drift):
    
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
        peak_position   = index to level off the velocity [-]
        E_drift         = drift efficiency                [-]
        
    KEYWORDS:
        nogrowth        = true: particle size fixed to a0 [False]
    
    RETURNS:
        time     = snap shot time           (nt,nr)       [s]
        solution = dust surface density     (nt,nr)       [g cm^-2]
        v_bar    = dust velocity            (nt,nr)       [cm s^-1]
        v_0      = small dust velocity      (nt,nr)       [cm s^-1]
        v_1      = large dust velocity      (nt,nr)       [cm s^-1]
        a_dr     = drift size limit         (nt,nr)       [cm]
        a_fr     = fragmentation size limit (nt,nr)       [cm]
        a_df     = drift-ind. frag. limit   (nt,nr)       [cm]
        a_t      = the time dependent limit (nt,nr)       [cm]
    """
    #
    # constants
    #
    plotting  = 0
    from constants import AU,year
    #
    # some setup
    #
    n_r    = len(x)
    n_t    = len(time)
    
    g    = ones(n_r)
    K    = zeros(n_r)
    L    = zeros(n_r)
    flim = ones(n_r)
    
    A0 = zeros(n_r)
    B0 = zeros(n_r)
    C0 = zeros(n_r)
    D0 = zeros(n_r)
    #
    # setup
    #
    t             = time[0]
    solution      = zeros([n_t,n_r])
    solution[0,:] = sig_d
    v_bar         = zeros([n_t,n_r])
    Diff          = zeros([n_t,n_r])
    v_0           = zeros([n_t,n_r])
    v_1           = zeros([n_t,n_r])
    a_t           = zeros([n_t,n_r])
    a_df          = zeros([n_t,n_r])
    a_fr          = zeros([n_t,n_r])
    a_dr          = zeros([n_t,n_r])
    u_in          = solution[0,:]*x
    it_old        = 1
    snap_count    = 0
    progress_bar(round((it_old-1)/(n_t-1)*100),'toy model running')
    #
    # save the velocity which will be used
    #
    res  = two_pop_velocity_nointerp(t,solution[0,:],x,sig_g,v_gas,T,alpha,m_star,a_0,V_FRAG,RHO_S,peak_position,E_drift,nogrowth=nogrowth)
    v_bar[0,:] = res[0]
    Diff[0,:]  = res[1]
    v_0[0,:]   = res[3]
    v_1[0,:]   = res[4]
    a_t[0,:]   = res[5]
    a_df[0,:]  = res[6]
    a_fr[0,:]  = res[7]
    a_dr[0,:]  = res[8]
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
        res   = two_pop_velocity_nointerp(t,u_in/x,x,sig_g,v_gas,T,alpha,m_star,a_0,V_FRAG,RHO_S,peak_position,E_drift,nogrowth=nogrowth)
        v     = res[0]
        D     = res[1]
        sig_g = res[2]
        v[0] = v[1]
        D[0] = D[1]
        D[-2:] = 0
        v[-2:] = 0
        #
        # to turn off diffusion:
        #
        #D = zeros(n_r)
        #
        # set up the equation
        #
        h    = sig_g*x
        #
        # do the update
        #
        #u_out = impl_donorcell_adv_diff_delta(n_r,x_1,D,v,g,h,K,L,flim,u_in,dt,0,0,1,1,1e-100*x_1(1),1e-100*x_1(end),1,A,B,C,D);
        u_out = impl_donorcell_adv_diff_delta(n_r,x,D,v,g,h,K,L,flim,u_in,dt,1,1,0,0,0,0,1,A0,B0,C0,D0)
        mask = abs(u_out[2:-1]/u_in[2:-1]-1)>0.05
        #
        # try variable time step
        #
        while any(u_out[2:-1][mask]/x[2:-1][mask]>=1e-30):
            dt = dt/10.
            if dt<1.0:
                print('ERROR: time step got too short')
                sys.exit(1)
            u_out = impl_donorcell_adv_diff_delta(n_r,x,D,v,g,h,K,L,flim,u_in,dt,1,1,0,0,0,0,1,A0,B0,C0,D0)
            mask = abs(u_out[2:-1]/u_in[2:-1]-1)>0.3
        #
        # update
        #
        u_in  = u_out[:]
        t     = t + dt
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
            solution[snap_count,:] = u_out/x
            v_bar[snap_count,:]    = v
            Diff[snap_count,:]     = D
            #
            # store the rest
            #
            v_0[snap_count,:]   = res[3]
            v_1[snap_count,:]   = res[4]
            a_t[snap_count,:]   = res[5]
            a_df[snap_count,:]  = res[6]
            a_fr[snap_count,:]  = res[7]
            a_dr[snap_count,:]  = res[8]

    progress_bar(100.,'toy model running')
    
    if plotting:
        #
        # plot the solution
        #
        widget.plotter(x=x/AU,
                       data=solution,
                       times=time/year,
                       xlog=1,
                       ylog=1,
                       xlim=[x[0]/AU,x[-1]/AU],
                       ylim=[1e-6,1e1],
                       xlabel='r [AU]',
                       i_start=0,
                       ylabel='$\Sigma_d$ [g cm $^{-2}$]')

    return [time,solution,v_bar,v_0,v_1,a_dr,a_fr,a_df,a_t]

def two_pop_model_run(x_1,a_0,timesteps_1,sigma_g_1,sigma_d_1,v_gas_1,T_1,alpha_1,m_star_1,T_COAG_START,V_FRAG,RHO_S,peak_position_1,E_drift,plotting=False,nogrowth=False):
    """
    this function evolves the two population model (all model settings
    are stored in two_pop_velocity). It returns the important parameters of
    the model.
    
    USAGE:
    [time,solution,v_bar]=two_pop_model(x_1,a_0,timesteps_1,sigma_g_1,s
    igma_d_1,v_gas_1,T_1,alpha_1,m_star_1,T_COAG_START,V_FRAG,RHO_S,peak_position_1)
    
    WHERE:
        x_1             = radial grid (nr)                [cm]
        a_0             = monomer size                    [cm]
        timesteps_1     = time of snapshots (nt)          [s]
        sigma_g_1       = gas  surface density (nt,nr)    [g cm^-2]
        sigma_d_1       = dust surface density (nt*nm,nr) [g cm^-2]
        v_gas_1         = gas velocity (nt,nr)            [cm/s]
        T_1             = temperature grid (nt,nr)        [K]
        alpha_1         = turbulence parameter (nt,nr)    [-]
        m_star_1        = stellar mass (nt)               [g]
        T_COAG_START    = time when coagulation starts    [s]
        V_FRAG          = fragmentation velocity          [cm s^-1]
        RHO_S           = internal density of the dust    [g cm^-3]
        peak_position_1 = index to level off the velocity [-]
        E_drift         = drift efficiency                [-]
    
    RETURNS:
        time     = snap shot time       (nt2,nr)        [s]
        solution = dust surface density (nt2,nr)        [g cm^-2]
        v_bar    = dust velocity        (nt2,nr)        [cm s^-1]
    """
    #
    # constants
    #
    plotting  = 0
    AU        = 1.496e13			# astronomical unit in cm
    year      = 31558149.54e0		# year in s
    #
    # some setup
    #
    n_r    = len(x_1)
    n_t    = len(timesteps_1)
    grains = shape(sigma_d_1)[0]/n_t
    
    g    = ones(n_r)
    K    = zeros(n_r)
    L    = zeros(n_r)
    flim = ones(n_r)
    
    A0 = zeros(n_r)
    B0 = zeros(n_r)
    C0 = zeros(n_r)
    D0 = zeros(n_r)
    
    sigma_d_total=zeros(shape(sigma_g_1))
    for i in arange(n_t):
        sigma_d_total[i,:] = sum(sigma_d_1[(i*grains):(i+1)*grains,:],0)
    #
    # setup
    #
    t             = T_COAG_START
    it0           = find(timesteps_1>=t)[0]
    t             = timesteps_1[it0]
    time          = zeros(n_t+1-it0)
    time[0]       = t
    solution      = zeros([n_t+1-it0,n_r])
    solution[0,:] = sigma_d_total[it0,:]
    v_bar         = zeros([n_t+1-it0,n_r])
    Diff          = zeros([n_t+1-it0,n_r])
    v_0           = zeros([n_t+1-it0,n_r])
    v_1           = zeros([n_t+1-it0,n_r])
    a_t           = zeros([n_t+1-it0,n_r])
    a_df          = zeros([n_t+1-it0,n_r])
    a_fr          = zeros([n_t+1-it0,n_r])
    a_dr          = zeros([n_t+1-it0,n_r])
    u_in          = solution[0,:]*x_1
    it_old        = it0+1
    snap_count    = 1
    progress_bar(round((snap_count-1)/(n_t-it0)*100),'toy model running')
    #
    # save the velocity which will be used
    #
    res  = two_pop_velocity(t,solution[0,:],x_1,timesteps_1,sigma_g_1,v_gas_1,T_1,alpha_1,m_star_1,a_0,T_COAG_START,V_FRAG,RHO_S,peak_position_1,E_drift,nogrowth=nogrowth)
    v_bar[0,:] = res[0]
    Diff[0,:]  = res[1]
    v_0[0,:]   = res[3]
    v_1[0,:]   = res[4]
    a_t[0,:]   = res[5]
    a_df[0,:]  = res[6]
    a_fr[0,:]  = res[7]
    a_dr[0,:]  = res[8]
    #
    # the loop
    #
    dt = Inf
    while t<timesteps_1[-1]:
        #
        # set the time step
        #
        dt = min(dt*10,timesteps_1[it_old]-t)
        if t != 0.0: dt = min(dt,t/200)
        if dt==0:
            print('ERROR:')
            print('t      = %g years'%(t/year))
            print('it_old = %g'%it_old)
            print('dt = 0')
            sys.exit(1)
        #
        # calculate the velocity
        #
        res   = two_pop_velocity(t,u_in/x_1,x_1,timesteps_1,sigma_g_1,v_gas_1,T_1,alpha_1,m_star_1,a_0,T_COAG_START,V_FRAG,RHO_S,peak_position_1,E_drift,nogrowth=nogrowth)
        v     = res[0]
        D     = res[1]
        sig_g = res[2]
        v[0] = v[1]
        D[0] = D[1]
        D[-2:] = 0
        v[-2:] = 0
        #
        # to turn off diffusion:
        #
        #D = zeros(n_r)
        #
        # set up the equation
        #
        h    = sig_g*x_1
        #
        # do the update
        #
        #u_out = impl_donorcell_adv_diff_delta(n_r,x_1,D,v,g,h,K,L,flim,u_in,dt,0,0,1,1,1e-100*x_1(1),1e-100*x_1(end),1,A,B,C,D);
        u_out = impl_donorcell_adv_diff_delta(n_r,x_1,D,v,g,h,K,L,flim,u_in,dt,1,1,0,0,0,0,1,A0,B0,C0,D0)
        mask = abs(u_out[2:-1]/u_in[2:-1]-1)>0.05
        #
        # try variable time step
        #
        while any(u_out[2:-1][mask]/x_1[2:-1][mask]>=1e-30):
            dt = dt/10.
            if dt<1.0:
                print('ERROR: time step got too short')
                sys.exit(1)
            u_out = impl_donorcell_adv_diff_delta(n_r,x_1,D,v,g,h,K,L,flim,u_in,dt,1,1,0,0,0,0,1,A0,B0,C0,D0)
            mask = abs(u_out[2:-1]/u_in[2:-1]-1)>0.3
        #
        # update
        #
        u_in  = u_out[:]
        t     = t + dt
        #
        # find out if we reached a snapshot
        #
        if t>=timesteps_1[it_old]:
            #
            # one more step completed
            #
            it_old     = it_old + 1
            snap_count = snap_count + 1
            #
            # notify
            #
            progress_bar(round((snap_count-1.)/(n_t-it0)*100.),'toy model running')
            #
            # save the data
            #
            solution[snap_count,:] = u_out/x_1
            v_bar[snap_count,:]    = v
            Diff[snap_count,:]     = D
            time[snap_count]       = t
            #
            # store the rest
            #
            v_0[snap_count,:]   = res[3]
            v_1[snap_count,:]   = res[4]
            a_t[snap_count,:]   = res[5]
            a_df[snap_count,:]  = res[6]
            a_fr[snap_count,:]  = res[7]
            a_dr[snap_count,:]  = res[8]

    progress_bar(100.,'toy model running')
    
    if plotting:
        #
        # plot the solution
        #
        widget.plotter(x=x_1/AU,
                       data=solution,
                       data2=sigma_d_total[it0:,:],
                       times=timesteps_1[it0:]/year,
                       xlog=1,
                       ylog=1,
                       xlim=[x_1[0]/AU,x_1[-1]/AU],
                       ylim=[1e-6,1e1],
                       xlabel='r [AU]',
                       i_start=0,
                       ylabel='$\Sigma_d$ [g cm $^{-2}$]')

        figure()
        semilogx(x_1/AU,(solution[-1,:]-sigma_d_total[-1,:])/sigma_d_total[-1,:]*100)
        xlim(x_1[0]/AU,x_1[-1]/AU)
        ylim(-100,100)
    return [time,solution,v_bar,v_0,v_1,a_dr,a_fr,a_df,a_t]

def two_pop_velocity_nointerp(t,sigma_d_t,x,sigma_g,v_gas,T,alpha,m_star,a_0,V_FRAG,RHO_S,peak_position,E_drift,nogrowth=False):
    """
    This model takes a snapshot of temperature, gas surface density and so on
    and calculates values of the representative sizes and velocities which are
    used in the two population model.
    
    USAGE:
    [v_bar,D,sigma_g_i,v_0,v_1,a_max_t,a_df,a_fr,a_dr] = ...
    two_pop_velocity_nointerp(t,sigma_d_t,x,sigma_g,v_gas,T,alpha,m_star,a_0,V_FRAG,RHO_S,peak_position,E_drift)
    
    WHERE:
        t            = time at which to calculate the values  [s]
        sigma_d_t    = current dust surface density array (nr)[g cm^-2]
        x            = nr radial grid points (nr)             [cm]
        timesteps    = times of the snapshots (nt)            [s]
        sigma_g      = gas surface density snapshots (nt,nr)  [g cm^-2]
        v_gas        = gas radial velocity (nt,nr)            [cm s^-1]
        T            = temperature snapshots (nt,nr)          [K]
        alpha        = turbulence parameter (nt,nr)           [-]
        m_star       = stellar mass (nt)                      [g]
        a_0          = monomer size                           [cm]
        T_COAG_START = time where coagulation starts          [s]
        V_FRAG       = fragmentation velocity                 [cm s^-1]
        RHO_S        = dust internal density                  [g cm^-3]
        E_drift      = drift efficiency                       [-]
    
    KEYWORD:
        nogrowth     = wether a fixed particle size is used   [False]
    
    RETURNS:
        v_bar        = the mass averaged velocity (nt,nr)         [cm s^-1]
        D            = t-interpolated diffusivity (nt,nr)         [cm^2 s^-1]
        sigma_g_i    = t-interpolated gas surface density (nt,nr) [g cm^-2]
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
    from constants import pi,k_b,mu,m_p,Grav
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
    v_dr[0:peak_position] = v_dr[peak_position-1]
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
    
    return [v_bar,D,sigma_g,v_0,v_1,a_max_t_out,a_df,a_fr,a_dr]

def two_pop_velocity(t,sigma_d_t,x_1,timesteps_1,sigma_g_1,v_gas_1,T_1,alpha_1,m_star_1,a_0,T_COAG_START,V_FRAG,RHO_S,peak_position_1,E_drift,nogrowth=False):
    """
    This model takes the full (n_t,n_r) grid of temperature, gas surface
    density and so on and calculates the time interpolated values of the
    representative sizes and velocities which are used in the two population
    model.
    
    USAGE:
    [v_bar,D,sigma_g_i,v_0,v_1,a_max_t,a_df,a_fr,a_dr] = ...
        two_pop_velocity(t,sigma_d_t,x_1,timesteps_1,sigma_g_1,v_gas_1,T_1,...
        alpha_1,m_star_1,a_0,T_COAG_START,V_FRAG,RHO_S,peak_position_1)
    
    WHERE:
        t            = time at which to calculate the values  [s]
        sigma_d_t    = current dust surface density array (nr)[g cm^-2]
        x_1          = nr radial grid points (nr)             [cm]
        timesteps_1  = times of the snapshots (nt)            [s]
        sigma_g_1    = gas surface density snapshots (nt,nr)  [g cm^-2]
        v_gas_1      = gas radial velocity (nt,nr)            [cm s^-1]
        T_1          = temperature snapshots (nt,nr)          [K]
        alpha_1      = turbulence parameter (nt,nr)           [-]
        m_star_1     = stellar mass (nt)                      [g]
        a_0          = monomer size                           [cm]
        T_COAG_START = time where coagulation starts          [s]
        V_FRAG       = fragmentation velocity                 [cm s^-1]
        RHO_S        = dust internal density                  [g cm^-3]
        E_drift      = drift efficiency                       [-]
        
    KEYWORD:
        nogrowth     = wether a fixed particle size is used   [False]    
        
    RETURNS:
        v_bar        = the mass averaged velocity (nt,nr)         [cm s^-1]
        D            = t-interpolated diffusivity (nt,nr)         [cm^2 s^-1]
        sigma_g_i    = t-interpolated gas surface density (nt,nr) [g cm^-2]
        v_0          = t-interpolated vel. of small dust (nt,nr)  [cm s^-1]
        v_1          = t-interpolated vel. of large dust (nt,nr)  [cm s^-1]
        a_max_t      = maximum grain size (nt,nr)                 [cm]
        a_df         = the fragmentation-by-drift limit (nt,nr)   [cm]
        a_fr         = the fragmentation limit (nt,nr)            [cm]
        a_dr         = the drift limit (nt,nr)                    [cm]
    """
    itest = 0
    #
    # set the fudge parameters
    #
    if itest >= 1:
        fudge_fr = 0.53
        fudge_dr = 0.62
    else:
        fudge_fr = 0.37
        fudge_dr = 0.55
    #
    # set some constants
    #
    pi          = 3.141593        # PI
    k_b       = 1.380658e-16    # Boltzmann constant in erg/K
    mu        = 2.3                # mean molecular mass in proton masses
    m_p       = 1.6726231e-24    # proton mass in g
    Grav      = 6.67259e-8        # gravitational constant in cm^3 g^-1 s^-2
    n_r       = len(x_1)
    #
    # find the time index and the interpolation parameter
    #
    it = find(timesteps_1>=t)
    if len(it) == 0:
        it = 0
    else:
        it = it[0]-1

    eps = (t-timesteps_1[it])/(timesteps_1[it+1]-timesteps_1[it])
    #
    # now interpolate gas surface density, alpha, T, v_gas, and m_star
    #
    sigma_g_i = (1-eps)*sigma_g_1[it,:] + eps*sigma_g_1[it+1,:]
    alpha_i   = (1-eps)*alpha_1[it,:]   + eps*alpha_1[it+1,:]
    T_i       = (1-eps)*T_1[it,:]       + eps*T_1[it+1,:]
    v_gas_i   = (1-eps)*v_gas_1[it,:]   + eps*v_gas_1[it+1,:]
    m_star_i  = (1-eps)*m_star_1[it]    + eps*m_star_1[it+1]
    #
    # calculate the pressure power-law index
    #
    P_o   = sigma_g_1[it,:]   * sqrt(Grav*m_star_i/x_1**3) * sqrt(k_b*T_1[it,:]  /mu/m_p)
    P_n   = sigma_g_1[it+1,:] * sqrt(Grav*m_star_i/x_1**3) * sqrt(k_b*T_1[it+1,:]/mu/m_p)
    gamma_o      = zeros(n_r)
    gamma_n      = zeros(n_r)
    gamma_o[1:n_r-1] = x_1[1:n_r-1]/P_o[1:n_r-1]*(P_o[2:n_r]-P_o[0:n_r-2])/(x_1[2:n_r]-x_1[0:n_r-2])
    gamma_n[1:n_r-1] = x_1[1:n_r-1]/P_n[1:n_r-1]*(P_n[2:n_r]-P_n[0:n_r-2])/(x_1[2:n_r]-x_1[0:n_r-2])
    gamma_o[0]       = gamma_o[1]
    gamma_n[0]       = gamma_n[1]
    gamma_o[-1]      = gamma_o[-2]
    gamma_n[-1]      = gamma_n[-2]
    #
    # time interpolate it
    #
    gamma_i   = (1-eps)*gamma_o + eps*gamma_n
    if nogrowth:
        a_max       = a_0*ones(n_r)
        a_max_t     = a_max
        a_max_t_out = a_max
        a_fr        = a_max
        a_dr        = a_max
        a_df        = a_max
    else:
        #
        # calculate the sizes
        #
        a_fr  = fudge_fr*2*sigma_g_i*V_FRAG**2/(3*pi*alpha_i*RHO_S*k_b*T_i/mu/m_p)
        a_dr  = fudge_dr/E_drift*2/pi*sigma_d_t/RHO_S*x_1**2*(Grav*m_star_i/x_1**3)/(abs(gamma_i)*(k_b*T_i/mu/m_p))
        N     = 0.5
        a_df  = fudge_fr*2*sigma_g_i/(RHO_S*pi)*V_FRAG*sqrt(Grav*m_star_i/x_1)/(abs(gamma_i)*k_b*T_i/mu/m_p*(1-N))
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
        o_k         = sqrt(Grav*m_star_i/x_1**3)
        tau_grow    = sigma_g_i/maximum(1e-100,sigma_d_t*o_k)
        a_max_t     = minimum(a_max,a_0*exp(minimum(709.0,(t-T_COAG_START)/tau_grow)))
        a_max_t_out = minimum(a_max_out,a_0*exp(minimum(709.0,(t-T_COAG_START)/tau_grow)))
    #
    # calculate the Stokes number of the particles
    #
    St_0 = a_0     * RHO_S/sigma_g_i*pi/2
    St_1 = a_max_t * RHO_S/sigma_g_i*pi/2
    #
    # calculate the velocities of the two populations:
    # First: gas velocity
    #
    v_0 = v_gas_i/(1+St_0**2)
    v_1 = v_gas_i/(1+St_1**2)
    #
    # Second: drift velocity
    #
    v_dr = k_b*T_i/mu/m_p/(2*o_k*x_1)*gamma_i
    #
    # level of at the peak position
    #
    v_dr[0:peak_position_1[it]] = v_dr[peak_position_1[it]-1]
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
    D = alpha_i * k_b*T_i/mu/m_p/o_k
    
    return [v_bar,D,sigma_g_i,v_0,v_1,a_max_t_out,a_df,a_fr,a_dr]


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
    D05=zeros(n_x)
    h05=zeros(n_x)
    rhs=zeros(n_x)
    #
    # calculate the arrays at the interfaces
    #
    for i in arange(1,n_x):
        D05[i] = flim[i] * 0.5 * (Diff[i-1] + Diff[i])
        h05[i] = 0.5 * (h[i-1] + h[i])
    #
    # calculate the entries of the tridiagonal matrix
    #
    for i in arange(1,n_x-1):
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
        for i in arange(1,n_x-1):
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