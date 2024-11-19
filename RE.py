import scipy.constants as const
import numpy as np
import scipy.constants as const
from unit_conversion import *
from scipy.integrate import cumtrapz

# Important Functions Related to REs #

def getConnorHastieField(ne,lnA):
    # Constants
    m = const.m_e # kg
    e = const.e # C 
    c = const.c # m/s
    return  e**3 * ne * lnA / (m * c**2 * 4 * np.pi * const.epsilon_0**2) # N/C, Connor hastie

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
    return const.m_e * const.c**2 * ( (1+1/E_star)**0.5 - 1) # J

def getInternalInductance(jp,r,R0):
    # Returns normalized and internal plasma inductance in 1-D circular geometry
    # jp = plasma current denisty [A/m**2]
    # r = minor radius [m]
    # R0 = major radius [m]
    B_p = 1/r[1:] * cumtrapz(const.mu_0 * r * jp, x=r) # Poloidal Magnetic Field [T]
    li = 2 * np.pi * np.trapz(B_p**2 * r[1:], x= r[1:]) / (np.pi * r[-1]**2 * B_p[-1]**2) # Normalized Inductance
    Lint = li * (const.mu_0 * R0 / (2)) # Henry
    return li, Lint

def getCoulombLog(ne,T):
    # Units in SI
    return 14.78e0-0.5*np.log(ne/1.0e20)+np.log(T/1.0e3)

def getTransSpitzerResistivity(lnA,T,Z):
    # T = electron temp. [eV]
    # Z = effective ionization
    return 1.03e-2 * Z * lnA * T**(-3/2) # [Ohm-cm]
    
    
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
        alpha = E_star # Field normalized by Conor-Hastie field, must be greater than 1 for Dreicer source
        alpha[np.where(alpha<1)]=1 + 1e-5
    else:
        alpha = np.max(np.array([E_star,1+1e-5]))
    
        
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
    
    T = T * const.eV # J
    vth = np.sqrt(1*T/m) # m/s
    fee = e**4 * ne * lnA / (4 * np.pi * const.epsilon_0**2) / (m**2 * vth**3) # collison freq. [s^-1]
    
    Ec = e**3 * ne * lnA / (m * c**2 * 4 * np.pi * const.epsilon_0**2) # N/C, Connor hastie
    ED = e**3 * ne * lnA / (T * 4 * np.pi * const.epsilon_0**2) # N/C, Dreicer field
    E = E_star * Ec
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
        alpha = E_star # Field normalized by Conor-Hastie field, must be greater than 1 for Dreicer source
        alpha[np.where(alpha<1)]=1 + 1e-5
    else:
        alpha = np.max(np.array([E_star,1+1e-5]))
    
    E = E_CH * alpha
    x = (np.abs(E)*const.e*T)/(E_CH*const.m_e*const.c**2)
    sd = ne*nu*x**(-3.0e0*(1.0e0+Zeff)/1.6e1) *np.exp(-1.0e0/(4*x)-np.sqrt((1.0e0+Zeff)/x))
    return sd

def getf_beta(W):
    # Returns the tritium beta energy distribution
    # W = electron energy in keV
    Wmax = 18.6 # [kJ]
    p = ((W*1e3*const.eV/(const.m_e*const.c**2)+1)**2 -1)**0.5 # v/c in terms of W [-]
    gamma = np.sqrt(p**2+1) # [-]
    WW = const.m_e * const.c**2 * gamma / (1.3 * const.eV)  # keV
    
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
    
    tau_T = 4500 * 24 * 3600 # [s], tritium half life
    Wc = getCriticalRunawayEnergy(E_star) # J
    Wc = Wc / const.eV * 1e-3 # keV
    
    # Load tritium beta spectrum
    data = np.loadtxt('./RE_Solver/data/trit_beta.csv',delimiter=',')
    E = data[:,0]
    f_beta = data[:,1] / 9.1885835
    # Analytical Spectrum 
    # E = np.linspace(0.1,18.6,200) # eV
    # f_beta = getf_beta(E)
    
    F_beta = np.trapz(f_beta[np.where(E>=Wc)],x=E[np.where(E>=Wc)]) # Fraction of beta electrons w energy > critical runaway energy
    return np.log(2) * nT / tau_T * F_beta # m**-3/s

def getComptonSource(E_star,ne,device='ITER'):
     # E_star = electric field / Connor-hastie Field
     # ne = total electron denisty n_free + n_bound [m**-3]
    # device = [ITER,SPARC]
    
    E_gamma = np.logspace(-2,1.477,1000) # Gamma energies [MeV]

    Wc = getCriticalRunawayEnergy(E_star) # J
    Wc = Wc / const.eV * 1e-6 # MeV
    

    # Scattering X-Section
    sigma_T = 6.6e-29 # Thomson x-section m**2
    theta_e = np.arccos(1 - const.m_e*const.c**2/(E_gamma * 1e6 * const.eV) * (Wc/E_gamma)/(1-(Wc/E_gamma))) # Critical Scattering angle [rad]
    x = (E_gamma * 1e6 * const.eV)  / (const.m_e * const.c**2)
    term1 = (x**2 - 2*x - 2) / x**3 * np.log((1 + 2*x) / (1 + x * (1 - np.cos(theta_e))))
    term2 = 1 / (2*x) * (1 / (1 + x * (1 - np.cos(theta_e)))**2 - 1 / (1 + 2*x)**2)
    term3 = -1 / x**3 * (1 - x - (1 + 2*x) / (1 + x * (1 - np.cos(theta_e))) - x * np.cos(theta_e))
    sigma =  (3 * sigma_T / 8) * (term1 + term2 + term3) # Compton x-section m**2
    sigma[np.where(np.isnan(sigma))] = 0.0

    if device == 'ITER':
        # Normalized Gamma Flux Spectrum [per MeV]
        z = (np.log(E_gamma) + 1.2) 
        Gamma = 1e18 * np.exp(-np.exp(-z) - z + 1) / 1.9724492150892259 # [#/m**2/s/MeV]
    elif device == 'SPARC':
        data = np.loadtxt('./RE_Solver/data/SPARC_Gamma_Flux.csv')
        from scipy.interpolate import interp1d
        f = interp1d(data[0,:],data[1,:],fill_value="extrapolate")
        Gamma = f(E_gamma) # [#/m**2/s/MeV]
        
    S =  ne * np.trapz(Gamma * sigma, x=E_gamma)
    return S

    

### Avalanche Growth Rate ###

def get_avalance_growth_rate_CH(E_star,ne,inv_aspect_ratio,lnA,Z):
    # From Chiu-Harvey 1998
    # All units in SI
    
    # Constants
    m = const.m_e # kg
    e = const.e # C 
    c = const.c # m/s
    
    Ec = e**3 * ne * lnA / (m * c**2 * 4 * np.pi * const.epsilon_0**2) # N/C, Connor hastie
   
    fp = 1 - 1.46 * np.sqrt(inv_aspect_ratio)
    A = E_star - 1 - 0.2 * (Z+1)/(Z+5)
    B = 1/(E_star**2 + 3*(Z+1)**2/(Z+5))
    return e * Ec * E_star / ( m * c *lnA) * np.sqrt(E_star * A * fp * B / (Z+5)) # per second

def get_avalance_growth_rate_RP(E_star,ne,inv_aspect_ratio,lnA,Z):
    # Rosenbluth-Putviski formual from Chiu-Harvey 1998
    # Note inverse aspect ratio = r / R0 where is is poloidal radius value from 0 to a
    # All units in SI
    
    # Constants
    m = const.m_e # kg
    e = const.e # C 
    c = const.c # m/s
    
    # Caluclate tau [s]
    
    Ec = e**3 * ne * lnA / (m * c**2 * 4 * np.pi * const.epsilon_0**2) # N/C, Connor hastie
    gamma_minus1 = e * Ec / (2 * m * c * lnA) * (E_star - 1)
    fp = 1 - 1.46 * np.sqrt(inv_aspect_ratio)
    g = 1 + 4 * (Z+1)**2 / (fp * (Z+5))
    H = g / (E_star**2 -1 + ((Z+5)*g/(fp*(Z+1)**2)))
    return gamma_minus1 * np.sqrt( 4 * fp / ((Z+5) * (1 - E_star**-2 + H))) # per second

def get_avalance_growth_rate_RP2(E_star,ne,inv_aspect_ratio,lnA,Z):
    # From Breizman 2019 / Rosenbluth 1997
    # All units in SI
    
    # Constants
    m = const.m_e # kg
    e = const.e # C 
    c = const.c # m/s
    
    # Caluclate tau [s]
    Ec = e**3 * ne * lnA / (m * c**2 * 4 * np.pi * const.epsilon_0**2) # N/C, Connor-Hastie
    tau =  m * c / (e * Ec) # s
    a = (1 + 1.46 * inv_aspect_ratio**0.5 + 1.72 * inv_aspect_ratio)**(-1)
    
    c1 = 1 /(tau * lnA) * np.sqrt(np.pi * a / (3*(Z+5))) * (E_star-1)
    c2 = 1 - E_star**(-1) 
    c3 = 4 * np.pi * (Z+1)**2
    c4 = 3 * a * (Z + 5) * (E_star**2 + 4/a**2 - 1)

    return c1 * (c2 + c3/c4)**(-1/2) # per second


