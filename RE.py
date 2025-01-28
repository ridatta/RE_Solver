import scipy.constants as const
import numpy as np
import scipy.constants as const
from unit_conversion import *
from scipy.integrate import cumtrapz
from weights import *

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
    B_p = 1/r[1:] * cumtrapz(const.mu_0 * r * jp, x=r) # Poloidal Magnetic Field [T]
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
    # Units in SI
    return 14.78e0-0.5*np.log(ne/1.0e20)+np.log(T/1.0e3)
    

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
#     data = np.loadtxt('./RE_Solver/data/trit_beta.csv',delimiter=',')
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
        data = np.loadtxt('./RE_Solver/data/SPARC_Gamma_Flux.csv')
        from scipy.interpolate import interp1d
        f = interp1d(data[0,:],data[1,:],fill_value="extrapolate")
        Gamma = f(E_gamma) # [#/m**2/s/MeV]
        
    if isinstance(Wc, np.ndarray):
        S = np.zeros_like(Wc)
        for ii in range(Wc.shape[0]):
            sigma = getScatteringXSection(E_gamma,Wc[ii]) # scattering x-section vs photon energy
            S[ii] =  ne * np.trapz(Gamma * sigma, x=E_gamma)
    else:
        sigma = getScatteringXSection(E_gamma,Wc) # scattering x-section vs photon energy
        S =  ne * np.trapz(Gamma * sigma, x=E_gamma) 
    return S

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
    ED0 = get
    (n0,lnA,T0) # N/C; Pre-TQ Dreicer Field
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

    return c1 * (c2 + c3/c4)**(-1/2) # per second


