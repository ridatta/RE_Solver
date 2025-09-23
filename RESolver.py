import numpy as np
from RE import *
import scipy.constants as const
from scipy.integrate import cumtrapz

def solve(constants,init_conditions,tmax=1e-3,CFL=0.4,RE_sources=[1,0,0,1],cre=0.1*3e8,nprint=500,verbose=True,RE_start=0,solveT=False):
    # constants = [rmax, R0, ne, T, Zeff, lnA, eta]
                # where rmax [m] = minor radius
                # R0 [m] = major radius
                # ne [per m cubed] = electron density, [1 x nx] np.array or [1x1] float
                # T [eV] = electron temperature, [1 x nx] np.array or [1x1] float
                # Zeff [-] = effective ionization, [1 x nx] np.array or [1x1] float
                # lnA [-] = Coulomb Logarithm, [1 x nx] np.array or [1x1] float
                # eta [Ohm-m] = resitivity, [1 x nx] np.array or [1x1] float
    # init_conditions = [jp0, jr0] [A/m**2] Initial plasma and RE current density
    # tmax = end time [s]
    # CFL = CFL Number 
    # RE_sources = [iDreicer, iTritBeta, iCompton, iAvalanche] if = 1 turns on the source, if 0 turns off the source
    # cre [m/s] = RE velocity
    # nprint = # of time steps to store results
    # verbose = if true print details
    # returns out = dictionary with results
    
    ####
    
    rmax, R0, ne, T, Zeff, lnA, eta = constants # Constants for the solver, all units in SI except temperature [eV]
    jp, jr = init_conditions[0].copy(), init_conditions[1].copy() # Initial plasma and RE current [A/m**2]
    n0 = 1.0 * ne
    
    ## Grid ##
    nx = jp.shape[0]
    dr = rmax / (nx-1) # [m], resolution
    r = np.linspace(1e-3,rmax,nx) # grid
    
    print('nx = %d\n'%nx)
    print('I0 [MA] = %1.3g\n'%(2*np.pi*np.trapz(jp*r,x=r)*1e-6))
    print('Ir (t = 0) [MA] = %1.3g\n'%(2*np.pi*np.trapz(jr*r,x=r)*1e-6))

    ## Runaway Settings ###
    iDreicer, iTritBeta, iCompton, iAvalanche = RE_sources
    print("iDreicer = %d,iTritBeta = %d, iCompton = %d, iAvalanche = %d\n"%(iDreicer, iTritBeta, iCompton, iAvalanche))
    print("Runaway Velocity [m/s]= %1.4g\n"%cre) # Runaway electron velocity [m/s]

    ## Numerical Settings ##
    U = np.max(eta/const.mu_0) / dr # resistive diffusion velocity, m/s
    # Calculate the Expected Primary Rate, [s^-1]
    Estar = eta*(jp-jr)/getConnorHastieField(ne,lnA)
    gamma = 1e-15
    if iDreicer:
        gamma = gamma + (1/ne * getDreicerSource(Estar,ne,lnA,Zeff,T))
    if iTritBeta:
        gamma = gamma + analy_beta_source(Estar)
    if iCompton:
        gamma = gamma + analy_compton_source(Estar)
    tau = 1/np.max(gamma) # [s]
    dt = np.min([dr * CFL / U,CFL*tau]) # Take the smaller time
    print("Time Step dt [ms] = %1.3g\n"%(dt*1e3))

    # Expected Seed Value #
    jseed = cre * const.e * ne * gamma * dt + jr # A/m**2

    print('Seed Ir(t=dt) [MA] = %1.3g\n'%(1e-6*2*np.pi*np.trapz(jseed*r,x=r)))
    
#     print('Initial E/Ed [-] = %1.3g\n'%(np.max(eta*(jp-jr))/getDreicerField(ne,lnA,T)))
#     print('Initial Dreicer rate [m^-3s^-1] = %1.3g\n'%getDreicerSource(np.max(Estar),ne,lnA,Zeff,T))

    #### Time Loop ####

    tnow, numsteps = 0, 0
    out = {}
    Ip, Ir, t = [], [], []
    li = []
    t.append(tnow)
    Ip.append(2*np.pi*np.trapz(jp*r,x=r)) # A
    Ir.append(2*np.pi*np.trapz(jr*r,x=r)) # A
    li.append(getInternalInductance(jp,r,R0)[0])
    

    while (tnow <tmax):
        E = eta * (jp - jr) # calculate electric field, N/C
        
        if tnow >= RE_start:
            E_star = E / getConnorHastieField(ne,lnA) # Normalized E-field
            S_primary = np.zeros_like(E)
            if iDreicer:
                S_primary = S_primary + getDreicerSource(E_star,ne,lnA,Zeff,T) # Dreicer Source, m**-3/s
            if iTritBeta:
                S_primary = S_primary + 0.5 * n0 * analy_beta_source(E_star) # Beta Source, m**-3/s
            if iCompton:
                S_primary =  S_primary + ne * analy_compton_source(E_star) # Compton Source, m**-3/s
            gamma_av = iAvalanche * get_avalance_growth_rate_RP2(E_star,ne,r/R0,lnA,Zeff) # Avalanche rate, per s
        else:
            gamma_av = np.zeros_like(E)
            S_primary = np.zeros_like(E)
            

        # avlanche operator used in Martin-Solis (2017)
#         alp = np.sqrt(3*(5+Zeff)/np.pi)
#         gamma_av =  iAvalanche * const.e * E / (const.m_e * const.c *lnA * alp) # Avalanche rate [per s]

        # Update Step #

        # Calculate Gradients 
        dEdr = np.gradient(E,r)
        d2Edr2 = np.gradient(dEdr,r)

        # Update 
        jp[1:-1] = jp[1:-1] + dt/const.mu_0 * (d2Edr2[1:-1] + 1/r[1:-1] * dEdr[1:-1])
        jr[1:-1] = jr[1:-1] +  dt * gamma_av[1:-1]  * jr[1:-1]  + dt * (cre * const.e) * S_primary[1:-1]
        

        # Update BCs
        jp[0] = jp[1] # No gradient
        jr[0] = jr[1] # No gradient
        jp[-1] = jr[-1] # No boundary E
        
        
        # Update Free Electron Density
        n_RE = np.abs(jr) / (const.e * cre) # per cubic m
        ne = n0 - n_RE # per cubic m 

        # Update time
        numsteps = numsteps+1
        tnow = tnow+dt
        

        # Print progress #
        if (numsteps % nprint == 0):
           # Calculate plasma & runaway current
            Ip.append(2*np.pi*np.trapz(jp*r,x=r)) # A
            Ir.append(2*np.pi*np.trapz(jr*r,x=r)) # A
            t.append(tnow)
            li.append(getInternalInductance(jp,r,R0)[0])
            # Re calculate T and eta
            if solveT:
                # Thermal eqm.
                kappa = 0.07 * 2.18122e+26 # particles/(m-s)
                chi =  kappa / n0
                alpha = eta / (n0 * const.eV * chi)
                j = jp - jr 
                F1 = cumtrapz(-alpha*r*j**2,x=r,initial=0)
                F2 = cumtrapz(F1/r,x=r,initial=0)
                Tnum = F2 + (T[-1] - F2[-1]) # new Temperature
                eta = eta * (Tnum/T)**(-3/2) # update eta
                T = Tnum # Update T
            
            # Print
            if verbose:
#                 Ed = getDreicerField(ne,lnA,T)
                print("step = %d, time [ms] = %1.2f, Ip [MA] = %1.2f, E/Ec [-] = %1.2f"
                      %(numsteps,tnow*1e3,Ip[-1]*1e-6,np.max(E_star)))

    out['Ip'], out['Ir'], out['t'] = Ip, Ir, t
    out['jp'], out['jr'] = jp, jr
    out['init_conditions'] = init_conditions
    out['constants'] = constants
    out['li'] = li
    out['T'] = T
    return out

