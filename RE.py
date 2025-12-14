import scipy.constants as const
import numpy as np
import scipy.constants as const
from unit_conversion import *
from scipy.integrate import cumulative_trapezoid
from weights import *
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

data_path = './data/' # Path to data folder

# Important Functions Related to REs #

def getConnorHastieField(ne,lnA):
    # Constants
    m = const.m_e # kg
    e = const.e # C 
    c = const.c # m/s
    return  e**3 * ne * lnA / (m * c**2 * 4 * np.pi * const.epsilon_0**2) # N/C, Connor hastie

    # ne = density [m**-3/s]
    # lnA = coulomb log
    # Constants
    m = const.m_e # kg
    e = const.e # C 
    c = const.c # m/s
    T = T * const.eV # J
    return  e**3 * ne * lnA / (T * 4 * np.pi * const.epsilon_0**2) # N/C, Connor hastie

def getDreicerField(ne,lnA,T):
    # ne = density [m**-3/s]
    # lnA = coulomb log
    # Constants
    m = const.m_e # kg
    e = const.e # C 
    c = const.c # m/s
    T = T * const.eV # J
    return  e**3 * ne * lnA / (T * 4 * np.pi * const.epsilon_0**2) # N/C, Connor hastie


def getCriticalRunawayEnergy(E_star):
    # E_star = electric field / Connor-hastie Field
    # ne = density [m**-3/s]
    # lnA = coulomb log
    
    if isinstance(E_star, np.ndarray):
        alpha = E_star.copy() # Field normalized by Conor-Hastie field, must be greater than 1 for Dreicer source
        alpha[np.where(alpha<1)]=1 + 1e-6
    else:
        alpha = np.max(np.array([E_star,1+1e-6]))
    
    return const.m_e * const.c**2 * ((1-alpha**(-1))**(-1/2)-1) # [J]

def getInternalInductance(jp,r,R0):
    # Returns normalized and internal plasma inductance in 1-D circular geometry
    # jp = plasma current denisty [A/m**2]
    # r = minor radius [m]
    # R0 = major radius [m]
    B_p = 1/r[1:] * cumulative_trapezoid(const.mu_0 * r * jp, x=r) # Poloidal Magnetic Field [T]
    li = 2 * np.pi * np.trapz(B_p**2 * r[1:], x= r[1:]) / (np.pi * r[-1]**2 * B_p[-1]**2) # Normalized Inductance
    Lint = li * (const.mu_0 * R0 / (2)) # Henry
    return li, Lint

def get_tau_ee(ne,T):
    # Returns electron electron collision time in [s]
    # ne = electron density in [per cubic m]
    # T = electron temp. in [eV]
    
    import scipy.constants as const

    e = const.e
    eps0 = const.epsilon_0
    me = const.m_e
    
    lnLambda = 14.9 - 0.5 * np.log(ne/1e20) + np.log(T/1e3)
    return 8 * np.sqrt(2) * np.pi * (e * T) ** (3/2) * eps0 ** 2 * np.sqrt(me) / (lnLambda * e ** 4 * ne) # s



## Other Useful Functions ##

def getTransSpitzerResistivity(lnA,T,Z):
    # T = electron temp. [eV]
    # Z = effective ionization
    return 1.03e-2 * Z * lnA * T**(-3/2) # [Ohm-cm]

def getCoulombLog(ne,T):
    # Units in SI (could implement another one)
#     return 14.78e0-0.5*np.log(ne/1.0e20)+np.log(T/1.0e3)
    return 14.6 + 0.5 * np.log(T / (ne/1e20)) # RE-Thermal Coul. log
    

def getZbarTF(z, a, rho, tev):
    """
    Approximation to the LTE TF model
    
    Parameters:
    z   : Nuclear charge [-]
    a   : Atomic mass [-]
    rho : Mass density [g/cc]
    tev : Electron temperature [eV]
    mp  : Proton mass [g]

    Returns:
    Zbar : Average ionization state
    ne   : Electron number density (/cc)
    """
    # Constants
    a1 = 0.003323467
    a2 = 0.97183224
    a3 = 9.26148e-05
    a4 = 3.1016524
    b0 = -1.762999
    b1 = 1.4317567
    b2 = 0.31546338
    c1 = -0.36666667
    c2 = 0.98333333
    alpha = 14.3139316
    beta = 0.66240046

    # Calculations
    mp = const.m_p # kg, proton mass
    M = a * mp
    ni = rho / M  # Ion number density (/cc)

    tzero = tev / (z ** (4 / 3))
    rho = np.minimum(rho, 10 / 1e3)
    r = rho / (z * a)
    tf = tzero / (1 + tzero)

    x1 = np.log(tzero)
    x2 = a4 * x1
    x1 = a2 * x1
    x1 = np.exp(x1)
    x2 = np.exp(x2)

    aa = a1 * x1 + a3 * x2
    arg = b0 + b1 * tf + b2 * tf ** 7
    b = -np.exp(arg)
    c = c2 + (c1 * tf)

    x1 = np.log(r)
    x2 = c * x1
    x1 = b * x1
    x1 = np.exp(x1)
    q1 = aa * x1
    x1 = c * np.log(q1)

    q = np.exp(x2) + np.exp(x1)
    cm = 1.0 / c
    x1 = cm * np.log(q)

    q = np.exp(x1)

    x1 = np.log(q)
    x2 = beta * x1

    x = alpha * np.exp(x2)

    f = x / (1.0 + x + (np.sqrt(1.0 + (2.0 * x))))
    Zbar = f * z
    ne = Zbar * ni

    return Zbar, ne

# ADAS Coronal Eqm. Ne Ionization Model

def getNeZ(Tq):
    data = np.loadtxt(data_path + 'Ne_Coronal.csv',delimiter=',')
    T = data[:,0]
    Z = data[:,1]

    f = interp1d(T, Z, fill_value='extrapolate')
    
    Zvals = f(Tq)
    if isinstance(Tq, np.ndarray):
        Zvals[np.where(Zvals > 10)] = 10
        Zvals[np.where(Zvals < 0) ] = 0
        return Zvals
    else:
        return min(Zvals,10)
        




    
## Primary Sources ##

def getDreicerSource(E_star,ne,lnA,Z,T):
    # Returns n_RE in [m^-3/s]
    # E_star = electric field / Connor-hastie Field
    # All units in SI except T [eV]
    # using formula in Breizmann 2019; Connor-Hastie (Relativistic) Formula
    
    # Constants
    m = const.m_e # kg
    e = const.e # C 
    c = const.c # m/s
    
    T = T * const.eV # J
    vth = np.sqrt(T/m) # m/s
    
    if isinstance(E_star, np.ndarray):
        alpha = E_star.copy() # Field normalized by Conor-Hastie field, must be greater than 1 for Dreicer source
        alpha[np.where(alpha<1)]=1 + 1e-6
    else:
        alpha = np.max(np.array([E_star,1+1e-6]))
    
        
    ED = e**3 * ne * lnA / (T * 4 * np.pi * const.epsilon_0**2) # N/C, Dreicer field
    Ec = e**3 * ne * lnA / (m * c**2 * 4 * np.pi * const.epsilon_0**2) # N/C, Connor hastie
    E = Ec * alpha # Electric field

    term1 = (alpha * (Z + 1) - Z + 7)
    term2 = 2 * np.sqrt(alpha / (alpha - 1)) * (1 + Z) * (alpha - 2)
    h =  (1 / (16 * (alpha - 1))) * (term1 + term2)

    term1 = alpha - 0.5
    term2 = np.sqrt(alpha * (alpha - 1))
    lam =  8 * alpha * (term1 - term2)

    term1 = np.sqrt((Z + 1) * alpha**2 / (8 * (alpha - 1)))
    term2 = np.pi / 2 - np.arcsin(1 - 2 / alpha)
    gam =  term1 * term2
    
    # Compute the main expression
    C = 0.21 + 0.11*Z
    term1 = C * ne * (e**4 * ne * lnA) / (m**2 * vth**3) / (4 * np.pi * const.epsilon_0**2)
    term2 = (E / ED) ** (-h)
    exp_term = np.exp(-lam * (ED / (4 * E)) - (2 * (ED / E))**(1 / 2) * gam)
    
    return term1 * term2 * exp_term

def getDreicerSourceV2(E_star,ne,lnA,Z,T):
    # [m^-3/s]
    # E_star = electric field / Connor-hastie Field
    # All units in SI excpet T [eV]
    # Using Jayakumar & Fleishmann 1993; Non relativistic version
    
    # Constants
    m = const.m_e # kg
    e = const.e # C 
    c = const.c # m/s
    
    if isinstance(E_star, np.ndarray):
        alpha = E_star.copy() # Field normalized by Conor-Hastie field, must be greater than 1 for Dreicer source
        alpha[np.where(alpha<1)]=1 + 1e-6
    else:
        alpha = np.max(np.array([E_star,1+1e-6]))
    
    T = T * const.eV # J
    vth = np.sqrt(1*T/m) # m/s
    fee = e**4 * ne * lnA / (4 * np.pi * const.epsilon_0**2) / (m**2 * vth**3) # collison freq. [s^-1]
    
    Ec = e**3 * ne * lnA / (m * c**2 * 4 * np.pi * const.epsilon_0**2) # N/C, Connor hastie
    ED = e**3 * ne * lnA / (T * 4 * np.pi * const.epsilon_0**2) # N/C, Dreicer field
    E = alpha * Ec
    epsilon = E/ED
    c1 = -3 * (Z+1) / 16
    A = 0.21 + 0.11*Z
#     A = 1
    S = A * epsilon**c1 * np.exp(-0.25/epsilon - ((Z+1)/epsilon)**0.5)
    return  ne * fee * S

def getDreicerSourceM3DC1(E_star,ne,T,Zeff):
    
    lnA = 14.78e0-0.5*np.log(ne/1.0e20)+np.log(T/1.0e3)
    
    vth = np.sqrt(2*const.e*T/const.m_e)
    nu = ne*const.e**4*lnA/(4*np.pi*const.epsilon_0**2*const.m_e**2*vth**3)
    
    E_CH = getConnorHastieField(ne,lnA)
    
    if isinstance(E_star, np.ndarray):
        alpha = E_star.copy() # Field normalized by Conor-Hastie field, must be greater than 1 for Dreicer source
        alpha[np.where(alpha<1)]=1 + 1e-6
    else:
        alpha = np.max(np.array([E_star,1+1e-6]))
    
    E = E_CH * alpha
    x = (np.abs(E)*const.e*T)/(E_CH*const.m_e*const.c**2)
    sd = ne*nu*x**(-3.0e0*(1.0e0+Zeff)/1.6e1) *np.exp(-1.0e0/(4*x)-np.sqrt((1.0e0+Zeff)/x))
    return sd

## Partially Screened Dreicer from Hesslow + 2019 ##

def getDreicerRateHesslow(nD,nI,ZmaxI,ZI,EoED,T):
    
    # Returns the partially screened Dreicer source term [m^-3/s]
    # based on original MATLAB implementation in https://github.com/unnerfelt/dreicer-nn
    
    # nD = Diuterium denisty [per cubic m]
    # nI = Impurity Denisty  [per cubic m]
    # ZmaxI = Atomic no. of Impurity [-]
    # ZI = Impurity Charge [-]
    # EoED = Normalized Electric Field [-]
    # T = Temp. [eV]
    
    nfree = nD * 1 + nI * ZI # free electron density
    ntot = nD * 1 + nI * ZmaxI # total electron density
    Zeff = (nD * 1**2 + nI * ZI**2) / nfree # effective ioniztaion
    Zeff0 = (nD * 0 + nI * (ZmaxI**2-ZI**2)) / ntot
    Z0oZ = (nD * 1 + nI * ZI / ZmaxI) / ntot
    ZZ0 = (nD * 1 + nI * ZI * ZmaxI) / ntot
    lnnfree = np.log(nfree)
    nfreeontot = nfree / ntot
    
    # Input to NN
    X = np.vstack([Zeff, Zeff0, Z0oZ, ZZ0, lnnfree, nfreeontot, EoED, np.log(T / 510998.946)])
    
    # Load NN weights and construct structure
    weights = [W1.T, W2.T, W3.T, W4.T, W5]
    biases = [b1, b2, b3, b4, b5]
    biases = [b.reshape(b.shape[0],1) for b in biases]
    m_nodes = [biases[ii].shape[0] for ii in range(len(biases))]
    m_nodes = [W1.shape[1]] + m_nodes
    
    G = forward_nn_rotated(X,m_nodes,weights,biases)
    G = G * 4/(3 * np.sqrt(np.pi)) # Normalize units.
    
    tau = get_tau_ee(nfree,T)
    
    return nfree * G / tau # [per cubic m / s]

def forward_nn_rotated(X,m_nodes,weights,biases):
    # X = input [n_input x num_instances]
    # m_nodes = no. of nodes per layer; has length = num. of deep layers + 2 (gor input & output layers)

    n_layers = len(m_nodes) # Number of layers
    
    # Normalize 
    i_std = np.array([4.96121897262, 5.58308339503, 0.195384621847, 4.98281996514, 2.45185472351, 0.304090526425, 0.0345813696948, 2.28862902169])
    i_mean = np.array([7.1561340244, 9.20374749346, 0.105979895644, 7.08910012241, 47.8203654981, 0.536347788327, 0.0699809427188, -9.4087288982])
    o_std = 18.5581667797
    o_mean = -23.1854797433
    
    X = (X - i_mean.reshape(i_mean.shape[0],1)) / i_std.reshape(i_std.shape[0],1) # Nomalize
    

    # Generate output
    Xnow = X.copy()
    for ii in range(n_layers-2):
        Xnow = np.tanh(biases[ii] + np.dot(weights[ii],Xnow)) # output of current layer
    
    Yout = biases[-1] + np.dot(weights[-1],Xnow)
    
    # De-normalize output
    Yout = np.exp(Yout * o_std + o_mean)
    
    return np.squeeze(Yout)

##### End of Partially Screened Dreicer from Hesslow + 2019 #####

def getf_beta(W):
    # Returns the tritium beta energy distribution
    # W = electron energy in keV
    Wmax = 18.6 # [kJ]
    w = (W * 1e3 * const.eV) / (const.m_e * const.c**2) # [-]
    p = ((w+1)**2-1)**(1/2)
    gamma = np.sqrt(p**2+1) # [-]
    WW = const.m_e * const.c**2 * gamma / (1e3 * const.eV)  # keV
    
    # Calculate F #
    Zf = 2
    alpha = 1/137
    beta = p / gamma
    c0 = 2 * np.pi * alpha * Zf
    c1 = beta * (1-np.exp(-c0/beta))
    F = c0 / c1
    
    f_ana = F * p * WW * (Wmax - W)**2
    
    # normalize 
    tot = np.trapz(f_ana,W)
    return 1/tot * f_ana

def getTritBetaDecay(nT,E_star):
    # E_star = electric field / Connor-hastie Field
    # Wc = critical runaway energy [keV]
    # nT = tritium denisty [m**-3]
    
    if isinstance(E_star, np.ndarray):
        alpha = E_star.copy() # Field normalized by Conor-Hastie field, must be greater than 1 for Dreicer source
        alpha[np.where(alpha<1)]=1 + 1e-6
    else:
        alpha = np.max(np.array([E_star,1+1e-6]))
    
    tau_T = 4500 * 24 * 3600 # [s], tritium half life
    Wc = getCriticalRunawayEnergy(alpha) # J
    Wc = Wc / const.eV * 1e-3 # keV
    
    # Load tritium beta spectrum
#     data = np.loadtxt(data_path + 'trit_beta.csv',delimiter=',')
#     E = data[:,0]
#     f_beta = data[:,1] / 9.1885835

    # Analytical Spectrum 
    E = np.linspace(0.1,18.6,200) # eV
    f_beta = getf_beta(E)
    
    if isinstance(Wc, np.ndarray): # Get Fraction of beta electrons w energy > critical runaway energy
        F_beta = np.zeros_like(Wc)
        for ii in range(Wc.shape[0]):
            F_beta[ii] = np.trapz(f_beta[np.where(E>=Wc[ii])],x=E[np.where(E>=Wc[ii])])
    else:
         F_beta = np.trapz(f_beta[np.where(E>=Wc)],x=E[np.where(E>=Wc)])
    
    return np.log(2) * nT / tau_T * F_beta # m**-3/s



def analy_beta_source(E_star):
    """
    Returns the beta source rate in per second. Uses an analytical fit to the integral for faster computation.
    Multiply with tritium density to get the generation rate in per cubic meter per second.

    Parameters:
        E_star (float): E-field normalized by Connor-Hastie field

    Returns:
        float: Beta source rate in per second
    """
    # Fit parameters
    A = 2.09023917e-09
    mu = -3.80424810
    sig = 6.81315848

    # Critical Runaway energy in keV
    Wc = getCriticalRunawayEnergy(E_star) / const.eV * 1e-3  # keV
    
    if isinstance(Wc, np.ndarray):
        outval = A * np.exp(-((Wc - mu) ** 2) / (2 * sig ** 2))
        outval[np.where(Wc > 18.6)] = 0.0
        outval[np.where(Wc < 0.1)] = 1.8e-9  # Maximum Rate, per s
    else:
        if Wc > 18.6:
            outval = 0.0
        elif Wc < 0.1:
            outval = 1.8e-9  # Maximum Rate
        else:
            outval = A * np.exp(-((Wc - mu) ** 2) / (2 * sig ** 2))

    return outval



def analy_compton_source(E_star):
    """
    Returns the Compton rate in per second. Uses an analytical fit to the integral for faster computation.
    Multiply with total electron density to get the rate in per cubic meter per second.

    Parameters:
        E_star (float): E-field normalized by Connor-Hastie field

    Returns:
        float: Compton rate in per second
    """
    # Critical Runaway energy in keV
    Wc = getCriticalRunawayEnergy(E_star) / const.eV * 1e-3  # keV
    x = np.log(Wc)
    
    # Fit parameters for SPARC
    A = 6.00578227e-01
    L = 1.25488546e+00
    x0 = 4.44639341e+00
    c = 2.12986727e-11
    B = -2.17893889e-11
    
    if isinstance(Wc, np.ndarray):
        outval = B * np.tanh(A * (x - x0) / L) + c
        outval[np.where(Wc > 10**4.5)] = 0.0
        outval[np.where(Wc < 0.1)] = 4.3e-11  # Maximum Rate, per s
    else:
        if Wc < 0.1:
            outval = 4.3e-11  # MAX rate for SPARC
        elif Wc > 10 ** 4.5:
            outval = 0.0
        else:
            outval = B * np.tanh(A * (x - x0) / L) + c

    return outval




def getComptonSource(E_star,ne,device='ITER'):
     # E_star = electric field / Connor-hastie Field
     # ne = total electron denisty n_free + n_bound [m**-3]
    # device = [ITER,SPARC]
    
    if isinstance(E_star, np.ndarray):
        alpha = E_star.copy() # Field normalized by Conor-Hastie field, must be greater than 1 for Dreicer source
        alpha[np.where(alpha<1)]=1 + 1e-6
    else:
        alpha = np.max(np.array([E_star,1+1e-6]))

    Wc = getCriticalRunawayEnergy(alpha) # J
    Wc = Wc / const.eV * 1e-6 # MeV

    E_gamma = np.logspace(-2,1.477,1000) # Gamma energies [MeV]
    if device == 'ITER':
        # Normalized Gamma Flux Spectrum [per MeV]
        z = (np.log(E_gamma) + 1.2) 
        Gamma = 1e18 * np.exp(-np.exp(-z) - z + 1) / 1.9724492150892259 # [#/m**2/s/MeV]
    elif device == 'SPARC':
        data = np.loadtxt(data_path + 'SPARC_Gamma_Flux.csv')
        f = interp1d(data[0,:],data[1,:],fill_value="extrapolate")
        Gamma = f(E_gamma) # [#/m**2/s/MeV]
    elif device == 'ARC-P':
        data = np.loadtxt(data_path + 'ARC_Prompt_Gamma_Flux.csv')
        f = interp1d(data[0,:],data[1,:],fill_value="extrapolate")
        Gamma = f(E_gamma) # [#/m**2/s/MeV]
    elif device == 'ARC-NP':
        data = np.loadtxt(data_path + 'ARC_NonPrompt_Gamma_Flux.csv')
        f = interp1d(data[0,:],data[1,:],fill_value="extrapolate")
        Gamma = f(E_gamma) # [#/m**2/s/MeV]
    
        
    if isinstance(Wc, np.ndarray):
        S = np.zeros_like(Wc)
        for ii in range(Wc.shape[0]):
            sigma = getScatteringXSection(E_gamma,Wc[ii]) # scattering x-section vs photon energy
            S[ii] =  np.trapz(Gamma * sigma, x=E_gamma)
    else:
        sigma = getScatteringXSection(E_gamma,Wc) # scattering x-section vs photon energy
        S =  np.trapz(Gamma * sigma, x=E_gamma) 
    return ne * S

def getScatteringXSection(E_gamma,Wc):
    # Wc in MeV
    # Output in m**2
    sigma_T = 6.6e-29 # Thomson x-section m**2
    val = 1 - const.m_e*const.c**2/(E_gamma * 1e6 * const.eV) * (Wc/E_gamma)/(1-(Wc/E_gamma))
    theta_e = np.ones_like(E_gamma) * float('nan')
    idx = np.argwhere(np.abs(val)<=1)
    theta_e[idx] = np.arccos(val[idx]) # Critical Scattering angle [rad]
    x = (E_gamma * 1e6 * const.eV)  / (const.m_e * const.c**2)
    term1 = (x**2 - 2*x - 2) / x**3 * np.log((1 + 2*x) / (1 + x * (1 - np.cos(theta_e))))
    term2 = 1 / (2*x) * (1 / (1 + x * (1 - np.cos(theta_e)))**2 - 1 / (1 + 2*x)**2)
    term3 = -1 / x**3 * (1 - x - (1 + 2*x) / (1 + x * (1 - np.cos(theta_e))) - x * np.cos(theta_e))
    sigma =  (3 * sigma_T / 8) * (term1 + term2 + term3) # Compton x-section m**2
    sigma[np.isnan(sigma)] = 0

    return sigma

def getHotTailSeed(n0,T0,lnA,E0,t0):
    # n0 = per m cubed; pre-TQ density 
    # T0 = eV, pre-TQ temp.
    # lnA = Coul. log
    # E0 = initial electro cfield [N/C]
    # t0 = TQ time [s]
    E0= E0 + 1e-8 # add a small value to prevent zeros
    vth = np.sqrt(T0*const.eV/const.m_e) # Pre-TQ electron thermal velocity
    nu0 = n0 * const.e**4 * lnA / (4 * np.pi * const.epsilon_0**2 * const.m_e**2 * vth**3) # per s, collison freq.
    ED0 = getDreicerField(n0,lnA,T0) # N/C; Pre-TQ Dreicer Field
    uc = (t0 * nu0 * (2 * np.log(ED0/E0) - 4/3*np.log(4/3*t0*nu0) - 5/3))**(1/3)
    nr = n0 * 2/np.sqrt(np.pi) * uc * np.exp(-uc**2) # per m cubed; runaway electron density
    return nr # per m cubed

### Avalanche Growth Rate ###

def get_avalance_growth_rate_CH(E_star,ne,inv_aspect_ratio,lnA,Z):
    # From Chiu-Harvey 1998
    # All units in SI
    
    # Constants
    m = const.m_e # kg
    e = const.e # C 
    c = const.c # m/s
    
    val = 1 + 0.2 * (Z+1)/(Z+5)
    if isinstance(E_star, np.ndarray):
        alpha = E_star.copy() # Field normalized by Conor-Hastie field
        alpha[np.where(alpha<val)]=val+1e-6
    else:
        alpha = np.max(np.array([E_star,val+1e-6]))
    
    Ec = e**3 * ne * lnA / (m * c**2 * 4 * np.pi * const.epsilon_0**2) # N/C, Connor hastie
   
    fp = 1 - 1.46 * np.sqrt(inv_aspect_ratio)
    A = alpha - 1 - 0.2 * (Z+1)/(Z+5)
    B = 1/(alpha**2 + 3*(Z+1)**2/(Z+5))
    return e * Ec * alpha / ( m * c *lnA) * np.sqrt(alpha * A * fp * B / (Z+5)) # per second

def get_avalance_growth_rate_RP(E_star,ne,inv_aspect_ratio,lnA,Z):
    # Rosenbluth-Putviski formual from Chiu-Harvey 1998
    # Note inverse aspect ratio = r / R0 where is is poloidal radius value from 0 to a
    # All units in SI
    
    # Constants
    m = const.m_e # kg
    e = const.e # C 
    c = const.c # m/s
    
    if isinstance(E_star, np.ndarray):
        alpha = E_star.copy() # Field normalized by Conor-Hastie field, must be greater than 1 for Dreicer source
        alpha[np.where(alpha<1)]=1 
    else:
        alpha = np.max(np.array([E_star,1]))
    
    # Caluclate tau [s]
    
    Ec = e**3 * ne * lnA / (m * c**2 * 4 * np.pi * const.epsilon_0**2) # N/C, Connor hastie
    gamma_minus1 = e * Ec / (2 * m * c * lnA) * (alpha - 1)
    fp = 1 - 1.46 * np.sqrt(inv_aspect_ratio)
    g = 1 + 4 * (Z+1)**2 / (fp * (Z+5))
    H = g / (alpha**2 -1 + ((Z+5)*g/(fp*(Z+1)**2)))
    return gamma_minus1 * np.sqrt( 4 * fp / ((Z+5) * (1 - alpha**-2 + H))) # per second

def get_avalance_growth_rate_RP2(E_star,ne,inv_aspect_ratio,lnA,Z):
    # From Breizman 2019 / Rosenbluth 1997
    # All units in SI
    
    if isinstance(E_star, np.ndarray):
        alpha = E_star.copy() # Field normalized by Conor-Hastie field, must be greater than 1 for Dreicer source
        alpha[np.where(alpha<1)]=1 
    else:
        alpha = np.max(np.array([E_star,1]))
    
    # Constants
    m = const.m_e # kg
    e = const.e # C 
    c = const.c # m/s
    
    # Caluclate tau [s]
    Ec = e**3 * ne * lnA / (m * c**2 * 4 * np.pi * const.epsilon_0**2) # N/C, Connor-Hastie
    tau =  m * c / (e * Ec) # s
    a = (1 + 1.46 * inv_aspect_ratio**0.5 + 1.72 * inv_aspect_ratio)**(-1)
    
    c1 = 1 /(tau * lnA) * np.sqrt(np.pi * a / (3*(Z+5))) * (alpha-1)
    c2 = 1 - alpha**(-1) 
    c3 = 4 * np.pi * (Z+1)**2
    c4 = 3 * a * (Z + 5) * (alpha**2 + 4/a**2 - 1)
    
    
    return const.e / (const.m_e * const.c *lnA) * Ec * (alpha - 1) / np.sqrt(5 + Z) # per second

#     return c1 * (c2 + c3/c4)**(-1/2) # per second

# THIS IS THE HESSLOW AVALANCHING MODEL #

def calcDelta(Eceff,nuD0,nuS1,tauRadInv,phib1,phib2,pc0):
    return (nuD0/nuS1**2)*(nuD0*tauRadInv/Eceff+phib1+phib2*np.log(pc0))

def calcEceff(delta,nuS0,nuS1,nuD1,nuD0,pc0):
    return nuS0 + nuS1*( (1+nuD1/nuD0)*np.log(pc0) + np.sqrt(2*delta+1))

def getEc_eff(Z,Z0,nj,T,B,element='Ne'):
    
    # Returns the effective critical electric field and Ectot with Ne or Ar impurities in V/m
    # INPUTS:
    #       Z   : atomic number of each ion species
    #       Z0  : net charge of plasma for each ion species
    #       nj  : density of each ion species [m^(-3)]
    #       Z, Z0 and nj must be arrays of length nSpecies
    #       T   : temperature [eV]  (for calculation of the Coulomb logarithm)
    #       B   : magnetic field [T]
    #       element: "Ne" or "Ar"
   
    # pick the element
    if element == 'Ne':
        ln_aBar = np.array([4.7064, 4.6097, 4.5028, 4.3858, 4.2638, 4.1266, 3.9568, 3.6810, 3.1777, 3.1278])
        ln_I    = np.array([-8.2227, -8.0370, -7.8614, -7.6837, -7.4994, -7.2788, -6.9808, -6.5976, -5.8933, -5.8320])
    elif element == 'Ar':
        ln_aBar = np.array([4.5677, 4.5007, 4.4289, 4.3513, 4.2698, 4.1815, 4.0829, 3.9748,
                            3.8511, 3.7882, 3.7203, 3.6436, 3.5579, 3.4525, 3.3093, 3.0503, 2.5578, 2.5332])
        ln_I    = np.array([-7.9050, -7.7532, -7.6076, -7.4626, -7.3178, -7.1665, -7.0055, -6.8020,
                            -6.5538, -6.4647, -6.3644, -6.2465, -6.1070, -5.9219, -5.6535, -5.3213, -4.6937, -4.6598])
    else:
        raise ValueError("element must be 'Ne' or 'Ar'")
    
    ne_free = np.sum(Z0 * nj) # Free electron density, per cubic m 
    ntot = np.sum(Z * nj) # Bound + Free electron density, per cubic m 
    Zeff = np.sum((Z0**2 * nj))/ne_free # Effective Ionization
    Zfulleff = np.sum(Z**2 * nj)/ne_free 
    lnLStar = 14.6 + 0.5 * np.log(T / (ne_free/1e20)) # Coul. Log
    Ne = Z - Z0 # No. of bound electrons

    ## calculate the constants nuD0, nuD1, nuS0, nuS1
    nuD0 = (1 + Zeff) - 2/3 * np.sum(Ne**2 * nj)/ne_free /lnLStar
    nuD1 = Zfulleff/lnLStar
    nuS0 = 1
    nuS1 = (1 + 3 * np.sum(Ne * nj)/ne_free)/(2*lnLStar)
    
    for i in range(1,Z.shape[0]):
        nuD0 = nuD0 + (Z[i]**2 - Z0[i]**2) * ln_aBar[Z0[i]] * (nj[i]/ne_free) / lnLStar
        nuS0 = nuS0 + Ne[i] * (-ln_I[Z0[i]]-1) * (nj[i]/ne_free) / lnLStar

    #### Radiation Terms ####

    # synchrotron radiation term
    tauRadInv =  B**2/(ne_free/1e20)/(15.44*lnLStar);

    # bremsstrahlung term
    ALPHA = 1/137.036;
    bremsprefactor = ALPHA * Zfulleff/lnLStar;
    phib1 = bremsprefactor*0.35;
    phib2 = bremsprefactor*0.20;

    pc0 = nuD0/(2*nuS1);

    Ectot = ntot/ne_free # in units of E_c
    Eceff0 = Ectot
    delta0 = calcDelta(Eceff0,nuD0,nuS1,tauRadInv,phib1,phib2,pc0)
    delta1 = calcDelta(calcEceff(delta0,
                                nuS0,nuS1,nuD1,nuD0,pc0),
                       nuD0,nuS1,tauRadInv,phib1,phib2,pc0)

    EceffOverEctot = calcEceff(delta1,nuS0,nuS1,nuD1,nuD0,pc0)/Ectot
    Ectot =  ntot *  const.e**3  * lnLStar /(4 * np.pi * const.epsilon_0**2 * const.m_e * const.c**2);
    
    return Ectot * EceffOverEctot, Ectot


def get_nuD_nuS(p,Z,Z0,nj,T,element='Ne'): # default Ne
    # Outputs the deflection, slowing down freq (Hesslow 2018)
    # p = normalized momentum
    # Z = max. ionization
    # Z0 = ionization level
    # nj = density [per cubic m]
    # T = temperature [eV]
    # element = 'Ne' or 'Ar'
    
    # Constants
    if element == 'Ne':
        ln_aBar = np.array([4.7064, 4.6097, 4.5028, 4.3858, 4.2638, 4.1266, 3.9568, 3.6810, 3.1777, 3.1278])
        ln_I    = np.array([-8.2227, -8.0370, -7.8614, -7.6837, -7.4994, -7.2788, -6.9808, -6.5976, -5.8933, -5.8320])
    elif element == 'Ar':
        ln_aBar = np.array([4.5677, 4.5007, 4.4289, 4.3513, 4.2698, 4.1815, 4.0829, 3.9748,
                            3.8511, 3.7882, 3.7203, 3.6436, 3.5579, 3.4525, 3.3093, 3.0503, 2.5578, 2.5332])
        ln_I    = np.array([-7.9050, -7.7532, -7.6076, -7.4626, -7.3178, -7.1665, -7.0055, -6.8020,
                            -6.5538, -6.4647, -6.3644, -6.2465, -6.1070, -5.9219, -5.6535, -5.3213, -4.6937, -4.6598])
    else:
        raise ValueError("element must be 'Ne' or 'Ar'")
    
    
    aBar_Ne = np.exp(ln_aBar)
    I_Ne = np.exp(ln_I)
    k = 5
    
    # Other Quantities
    ne_free = np.sum(Z0 * nj) # Free electron density, per cubic m 
    Zeff = np.sum((Z0**2 * nj))/ne_free # Effective Ionization
    Ne = Z - Z0 # No. of bound electrons

    # Everything is a function of momentum p
    gamma = np.sqrt(1+p**2)
    beta = p**2 / (1 + p**2)
    
    # Coulomb Logs (also functions of p)
    PT = np.sqrt(2 * T / const.m_e / const.c**2) # Thermal velocity
    lnAc = 14.6 + 0.5 * np.log(T / (ne_free/1e20)) # RE-Thermal Coul. log
    lnA0 = 14.9 - 0.5 * np.log(ne_free/1e20) + np.log(T/1e3) # Thermal-Thermal Coul. Log
    lnAee = lnA0 + 1/5 * np.log(1 + (2*(gamma-1)/PT**2)**2.5)
    lnAei = lnA0 + 1/5 * np.log(1 + (2*p/PT)**5)
    # lnAee = lnAc + np.log(np.sqrt(gamma-1))
    # lnAei = lnAc + np.log(np.sqrt(2) * p)
    # lnA = 14.6 + 0.5 * np.log(T / (1e20/1e20))
    
    # Caluclate nuS and nuD (Bandaru + 2024)

    nuS = lnAee
    nuD = lnAee + Zeff * lnAei
    for j in range(1,Z.shape[0]):
        hj = p * np.sqrt(gamma-1) / I_Ne[Z0[j]]
        F1j = 1/k * np.log(1+hj**k) - beta**2
        pabar1p5 = (p*aBar_Ne[Z0[j]])**1.5
        nuS = nuS + (nj[j] / ne_free)  * Ne[j] * F1j
        nuD = nuD + (nj[j] / ne_free) * ( 
                      (2/3 * (Z[j]**2 - Z0[j]**2)) * np.log(pabar1p5+1)
                    - (2/3 * Ne[j]**2 * pabar1p5 / (pabar1p5+1))
                    )
        
    nuS = nuS / lnAc
    nuD = nuD / lnAc
    
    return nuD, nuS

    
# Get fn
# This describes the function to solve critical momentum
def fn(pstar,E_star,Z,Z0,nj,T):
    # T [eV]
    # Pstar = normalized momentum
    # Estar = Epar / E_CH (E_CH = Connor Hastie Critical Field)
    nuD, nuS = get_nuD_nuS(pstar,Z,Z0,nj,T)
    return pstar * (E_star)**(0.5) - (nuS * nuD)**(1/4)


# Hesslow Avalanching Rate
def getHesslowAvalanchingRate(E_star,Z,Z0,nj,T,element,B=0): 
    # Returns the avalanching rate in per second
    # E_star [-] = Parallel E-field / E_CH (E_CH = Connor Hastie Electric field)
    # Z = max. ionization; first element of array is deuterium, next is Neon
    # Z0 = ionization level; first element of array is 1, next is Neon ionization
    # nj = density of each species [per cubic m]
    # T = temperature [eV]
    # element = either 'Ar' or 'Ne'

    ne_free = np.sum(Z0 * nj) # Free electron density, per cubic m 
    ntot = np.sum(Z * nj) # Bound + Free electron density, per cubic m 
    lnAc = 14.6 + 0.5 * np.log(T / (ne_free/1e20)) # RE-Thermal Coul. log
    Ec = getConnorHastieField(ne_free, lnAc)  # [V/m]
    Epar = Ec * E_star # Parallel Electric Field V/m
    if ne_free == ntot: # test to handle fully ionized case
        Zeff = np.sum((Z0**2 * nj)) / ne_free
        Eceff = 1.0 * Ec
        nuD = 1 + Zeff
        nuS = 1
    else: 
        Eceff = getEc_eff(Z,Z0,nj,T,B,element)[0] # Effective E-field, V/m
        if (Epar > Eceff):
            pstar = fsolve(fn, 1, args=(E_star,Z,Z0,nj,T))[0] # critical momentum [-]
            nuD, nuS =  get_nuD_nuS(pstar,Z,Z0,nj,T,element) # Slowing down and deflection freq [-]
            
    if (Epar > Eceff):
        gamma = (const.e / (const.m_e * const.c *lnAc) 
                     * (ntot / ne_free) * (Epar - Eceff) / np.sqrt(4 + nuS * nuD)) # per second
    else:
        gamma = 0.0
    
    return gamma

# END OF HESSLOW AVALANCHING #


