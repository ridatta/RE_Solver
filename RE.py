import scipy.constants as const
import numpy as np
import scipy.constants as const
from unit_conversion import *
from scipy.integrate import cumulative_trapezoid
from weights import *
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

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


def getVolumetricComptonSource(E_star,ne,r,Y,E_gamma):
	# E_star = electric field / Connor-hastie Field
	# ne = total electron denisty n_free + n_bound [m**-3]
	# r = radial array [m]
	# Y = gamma emissivity [photons/m^3/s]
	# E_gamma = gamma energy [MeV]
	# NOTE that this assumes an emissivity with single energy, delta function, which could be updated

	if isinstance(E_star, np.ndarray):
		alpha 	= E_star.copy(); # Field normalized by Conor-Hastie field, must be greater than 1 for Dreicer source
		alpha[np.where(alpha<1)]=1 + 1e-6;
	else:
		alpha 	= np.max(np.array([E_star,1+1e-6]));
		
	Wc 		= getCriticalRunawayEnergy(alpha); # J
	Wc 		= Wc / const.eV * 1e-6; # MeV
	##print(Wc);
	
	Flux_outward 	= 1/r*cumulative_trapezoid(Y*r,x=r,initial=0); # flux of photons from inside to outside of flux surface [photons/m2/s]
	
	Flux_inward 	= np.zeros_like(r);
	for i in range(len(r)):
		Flux_inward[i] = 1/np.pi * np.trapz(Y[i:]*r[i:]/r[i]*np.arcsin(r[i]/r[i:]),x=r[i:]); # flux of photons from outside to inside of flux surface [photons/m2/s]
	
	Flux_total 	= Flux_outward + 2*Flux_inward; # because flux going in must also come out (assuming long mean free path of photon)
	
	sigma 		= np.zeros_like(Wc);
	for i in range(Wc.shape[0]):
	    if Wc[i] > E_gamma:	sigma[i] = 0;
	    else: 		sigma[i] = getScatteringXSection(np.array([E_gamma]),Wc[i]) # m^2, scattering x-section vs photon energy
	    
	S 		= Flux_total*sigma; # [electrons/s]
	
	return ne * S; # [electrons/m^3/s]
	

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
    elif device == 'SPARC-P':
        data = np.loadtxt(data_path + 'SPARC_Gamma_Flux.csv')
        f = interp1d(data[0,:],data[1,:],fill_value="extrapolate")
        Gamma = f(E_gamma) # [#/m**2/s/MeV]
    elif device == 'SPARC-NP':
        data = np.loadtxt(data_path + 'SPARC_Gamma_Flux.csv')
        f = interp1d(data[0,:],data[1,:],fill_value="extrapolate")
        Gamma = 1e-3 * f(E_gamma) # [#/m**2/s/MeV]
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

def get_screening_constants(Z_imp, element=None):
    """
    Can be used to defining all constants once at the beginning of the solver class
    Or just only call the relavent species 
    Z_imp : atomic number
    element : String name for certain elements (just to keep some existing structure) 
    """

    r_e = 5.11e5  # electron rest energy [eV]

    if element == 'Ne':
        ln_aBar = np.array([4.7064, 4.6097, 4.5028, 4.3858,
                            4.2638, 4.1266, 3.9568, 3.6810,
                            3.1777, 3.1278])

        ln_I = np.array([-8.2227, -8.0370, -7.8614, -7.6837,
                         -7.4994, -7.2788, -6.9808, -6.5976,
                         -5.8933, -5.8320])

    elif element == 'Ar':
        ln_aBar = np.array([4.5677, 4.5007, 4.4289, 4.3513, 4.2698, 4.1815, 4.0829, 3.9748,
                            3.8511, 3.7882, 3.7203, 3.6436, 3.5579, 3.4525, 3.3093, 3.0503, 2.5578, 2.5332])
        ln_I    = np.array([-7.9050, -7.7532, -7.6076, -7.4626, -7.3178, -7.1665, -7.0055, -6.8020,
                            -6.5538, -6.4647, -6.3644, -6.2465, -6.1070, -5.9219, -5.6535, -5.3213, -4.6937, -4.6598])

    elif element == 'Be':
        ln_aBar = np.log(np.array([159, 114, 67, 59]))
        ln_I = np.log((np.array([42.2, 76.9, 205.0, 240.2]) / r_e)**-1)

    elif element == 'C':
        ln_aBar = np.log(np.array([144, 118, 95, 70, 42, 39]))
        ln_I = np.log((np.array([65.9, 92.6, 134.8, 214.2,
                                  486.2, 539.5]) / r_e)**-1)

    elif element == 'W':
        ln_aBar = np.array([4.07753744, 4.06273922, 4.04771872, 4.03246916, 4.01698344, 4.00125414, 3.98527347, 3.96903326, 3.95252495, 3.93573953, 3.91866755, 3.90129904,
                            3.88362353, 3.86562997, 3.84730669, 3.8286414, 3.80962107, 3.79023195, 3.77045944, 3.75028808, 3.72970145, 3.70868208, 3.6872114, 3.66526959,
                            3.64283552, 3.61988658, 3.5963986, 3.57234564, 3.54769984, 3.52243125, 3.49650756, 3.47196645, 3.44680789, 3.42100001, 3.39450839, 3.36729583,
                            3.33932198, 3.31054301, 3.28091122, 3.25037449, 3.21887582, 3.19047635, 3.16124671, 3.13113691, 3.10009229, 3.06805294, 3.03495299, 3.00071982,
                            2.96527307, 2.92852352, 2.89037176, 2.86220088, 2.83321334, 2.80336038, 2.77258872, 2.74084002, 2.7080502, 2.67414865, 2.63905733, 2.60268969,
                            2.56494936, 2.53240712, 2.49986488, 2.46732264, 2.4347804, 2.40223816, 2.36969592, 2.33715368, 2.30461144, 2.2720692, 2.23952696, 2.20698472,
                            2.17444248, 2.14190024])
        ln_I = np.log((np.array([751, 775, 801, 827, 855, 885, 917, 950, 985, 1020, 1060, 1110, 1150, 1200, 1250, 1310, 1370, 1430, 1500, 1580, 1660, 1720, 1780, 1840, 1900,
                                 1970, 2050, 2130, 2210, 2300, 2400, 2500, 2610, 2730, 2850, 2990, 3140, 3300, 3470, 3660, 3870, 4100, 4350, 4630, 4940, 5290, 5680, 5960, 6270,
                                 6610, 6980, 7380, 7830, 8310, 8860, 9460, 1.01e4, 1.09e4, 1.17e4, 1.27e4, 1.38e4, 1.51e4, 1.66e4, 1.83e4, 2.04e4, 2.08e4, 2.14e4, 2.22e4,
                                 2.33e4, 2.50e4, 2.84e4, 3.64e4, 8.79e4, 8.91e4])/r_e)**-1)

    else:
        I0 = 14.99
        alpha = 1/137
        prefactor = (3/(2*alpha)) * (np.pi/3)**(1/3)

        if Z_imp < 1:
            raise ValueError("Z must be >= 1")

        ln_aBar = np.zeros(Z_imp + 1)
        ln_I = np.zeros(Z_imp + 1)

        for Z0_state in range(1, Z_imp + 1):

            Ne = Z_imp - Z0_state

            if Ne > 0:
                aBar = prefactor * (Ne**(2/3)) / Z_imp
                ln_aBar[Z0_state] = np.log(aBar)
            else:
                ln_aBar[Z0_state] = 0.0

            ln_I[Z0_state] = np.log((I0 * Z0_state**2) / r_e)

    return ln_aBar, ln_I

def getEc_eff(Z,Z0,nj,T,B,ln_aBar,ln_I):
    
    # Returns the effective critical electric field and Ectot with Ne or Ar impurities in V/m
    # INPUTS:
    #       Z   : atomic number of each ion species
    #       Z0  : net charge of plasma for each ion species
    #       nj  : density of each ion species [m^(-3)]
    #       Z, Z0 and nj must be arrays of length nSpecies
    #       T   : temperature [eV]  (for calculation of the Coulomb logarithm)
    #       B   : magnetic field [T]
    #       ln_aBar, ln_I : use get_screening_constants()
    
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
    for i in range(1, Z.shape[0]):
        ln_a = ln_aBar[Z0[i]]
        lnI  = ln_I[Z0[i]]

        nuD0 += (Z[i]**2 - Z0[i]**2) * ln_a * (nj[i]/ne_free) / lnLStar
        nuS0 += Ne[i] * (-lnI - 1) * (nj[i]/ne_free) / lnLStar

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


def get_nuD_nuS(p,Z,Z0,nj,T,ln_aBar,ln_I): # default Ne
    # Outputs the deflection, slowing down freq (Hesslow 2018)
    # p = normalized momentum
    # Z = max. ionization
    # Z0 = ionization level
    # nj = density [per cubic m]
    # T = temperature [eV]

    k = 5

    aBar = np.exp(ln_aBar)
    Ibar = np.exp(ln_I)
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
        I_j = Ibar[Z0[j]]
        a_j = aBar[Z0[j]]

        hj = p * np.sqrt(gamma-1) / I_j
        pabar1p5 = (p * a_j)**1.5

        F1j = 1/k * np.log(1+hj**k) - beta**2
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
def fn(pstar,E_star,Z,Z0,nj,T,ln_aBar,ln_I):
    # T [eV]
    # Pstar = normalized momentum
    # Estar = Epar / E_CH (E_CH = Connor Hastie Critical Field)
    nuD, nuS = get_nuD_nuS(pstar,Z,Z0,nj,T,ln_aBar,ln_I)
    return pstar * (E_star)**(0.5) - (nuS * nuD)**(1/4)


# Hesslow Avalanching Rate
def getHesslowAvalanchingRate(E_star,Z,Z0,nj,T,ln_aBar,ln_I,B=0): 
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
        Eceff = getEc_eff(Z, Z0, nj, T, B, ln_aBar, ln_I)[0] # Effective E-field, V/m

        if (Epar > Eceff):
            pstar = fsolve(fn, 1,args=(E_star, Z, Z0, nj, T, ln_aBar, ln_I))[0] # critical momentum [-]
            nuD, nuS = get_nuD_nuS(pstar, Z, Z0, nj, T, ln_aBar, ln_I) # Slowing down and deflection freq [-]
            
    if (Epar > Eceff):
        gamma = (const.e / (const.m_e * const.c *lnAc) 
                     * (ntot / ne_free) * (Epar - Eceff) / np.sqrt(4 + nuS * nuD)) # per second
    else:
        gamma = 0.0
    
    return gamma

# END OF HESSLOW AVALANCHING #

### HOT TAIL FUNCTIONS ###

def getf(v, n0, vT0, tau):
    """
    Corrected Maxwellian with tau
    v: velocity (m/s)
    n0: density (m^-3)
    vT0: initial thermal speed (m/s)
    tau: no units
    returns:
    f : m^3 
    """
    prefactor = n0 / (np.pi**1.5 * vT0**3)
    expo = -((v**3/vT0**3) + 3*tau)**(2/3)
    return prefactor * np.exp(expo)

def getfm(v, n0, vT):
    """
    Basic one without tau
    v: velocity (m/s)
    n0: density (m^-3)
    vT: current thermal speed (m/s)
    returns:
    f : m^3 
    """
    prefactor = n0 / (np.pi**1.5 * vT**3)
    expo = -((v**2/vT**2))
    return prefactor * np.exp(expo)


def temperature(t, T0, Tf, t0):
    """
    Parameters:
    t: Time (s)
    T0: initial temp (eV)
    Tfinal: final temp (eV)
    t0: initial time (s)
    """
    return Tf + (T0 - Tf) * np.exp(-t / t0)

def getnu0(n, T0):
    """
    Parameters:
    T: temp (joules)
    vT: initial time (s)
    returns rate in s^-1
    """
    vT0 = np.sqrt(2 * T0 * const.e / const.m_e) # m/s
    lnL = 15 
    return n*const.e**4 * lnL/(4*np.pi * const.epsilon_0**2 * const.m_e**2 * vT0**3)

def getTau(nu0, t, t0):
    """
    t0: initial time (s)
    t: time (s)
    nu0: collision rate (s^-1)
    returns:
    tau (no unit) 
    """
    # commented line includes time delay; otherwise set to be 0
    #return nu0 * max(0.0, t - t0)
    return nu0 * max(0.0, t - 0)

def get_vc(T,E,ne):
    lnA = 15
    Ed = getDreicerField(ne,lnA,T) # V/m
    vc2 = (T * const.e / const.m_e) * (Ed / E)
    return np.sqrt(vc2)

# define the hot-tail function

def getHT(ne,T0,E0,T,t,t0=0):
    # ne [per cubic m] = electron density
    # T0 [eV] = initial temperature
    # T [eV] = temperature at time t
    # E0 [V/m] = initial E-field 
    # t [s] = time
    # t0 [s] = temp. decay time, usually used in caluclation of tau but DREAM sets this to 0
    
    # thermal velocities
    vT0 = np.sqrt(2 * T0 * const.e / const.m_e) # m/s
    vT = np.sqrt(2 * T * const.e / const.m_e) # m/s
    
    # Collision freq.
    nu0 = getnu0(ne, T0)
    
    # get the distribution function
    x = np.linspace(0, 50, int(5e4))      # x = (v / vT0)^2
    v = np.sqrt(x)*vT0
    fm = getfm(v, ne, vT) # cold maxwellian
    tau = getTau(nu0, t, t0)
    f = getf(v, ne, vT0, tau) # hot population
    ftot = fm + f # total distribution
    
    # Electric field
    E = E0 * (T/T0)**(-3/2)
    vc = get_vc(T,E,ne) # m/s
    
    # Now integrate from vc to inf
    idx = np.argwhere(v >= vc)
    idx = idx.flatten()
    return np.trapz(4 * np.pi * (v[idx]**2-vc**2) * ftot[idx],v[idx]) # per cubic m

def dHT_dt(n0, T0, Tf, E0, t, t0=0, dt=1e-6):
    """
    time derivative(hot tail rate)
    Parameters:
    n0: density [m^-3]
    T0: initial temperature [eV]
    Tf: final temperature [eV]
    E0: initial electric field [V/m]
    t: time [s]
    t0: temperature decay time [s]
    dt : time step [s]
    """

    T_plus  = temperature(t + dt, T0, Tf, t0)
    T_minus = temperature(t - dt, T0, Tf, t0)
    HT_plus = getHT(n0, T0, E0, T_plus,  t + dt, t0)
    HT_minus= getHT(n0, T0, E0, T_minus, t - dt, t0)

    #return (HT_plus - HT_minus) / (2 * dt)
    return max(0.0, (HT_plus - HT_minus) / (2 * dt))

def get_HT_rate(ne,T,t,T0,E0,nht,dt):
    # Returns the HT rate in [per cubuc m per s] and the new HT density [per cubic m]
    # Inputs:
    # ne = electron density [per cubic m]
    # T [eV] = temp.
    # t [s] = time
    # T0 [eV], E0 [V/m] = initial Temp. & E-field
    # nht = previous HT value [per cubic m]
    # dt = time step [s]
    nht_new = max(nht,getHT(ne,T0,E0,T,t))
    rate = (nht_new - nht) / dt
    return rate, nht_new



## End of Hottail functions ##
#########################################################

######################################################### 
###################  Zero-D Solver Class ################
#########################################################

class solver0D:
    def __init__(self):
        # Constants
        self.R0 = 1.85 # Major Radius, [m]
        self.a = 0.57 # Minor Radius, [m]
        self.ni = 3e20 # ion density in plasma fuel, per m**3
        self.L = 1.85e-6 # inductance [H]
        self.k = 1 # elongation
        self.Btor = 12.2 # Toroidal Field [Tesla]

        # IC
        self.I0 = 8.7e6 # Initial plasma current [ampere]
        self.Ir0 = 0 # Initial RE current [ampere]
        
        # Temp. evolution
        self.T0 = 5e3 # eV, initial temperature
        self.Tf = 10 # eV, final temperature
        self.t0 = 0.1e-3 # TQ time [s]
        
        # Impurities
        self.nD = 1e-11 # added deuterium
        self.n_imp = 1e-11 # impurity density
        self.Z_imp = 10 # impuurity atomic no.
        self.A_imp = 20 # impurity weight

        self.prescibe_nz_evolution = False
        self.Zeff_scale = 1.0

        # Vloop Settings
        self.V0 = 0.0 # Volts, pre TQ loop voltage
        self.Vf = 0.0 # Volts, post TQ loop voltage

        # Settings

        self.iDreicer = 1
        self.iTritium = 0
        self.iCompton = 0
        self.iAvalanche = 3

        self.tstart = 0.0 # s, start time
        self.tmax = 5e-3 # s, max. solution time
        self.idiag = False # stores detale doutput if true
        self.device = 'SPARC'


        
    def display(self):
        print("\n Constants:")
        print('Major radius R0 [m] = %1.3f'%(self.R0))
        print('Minor radius a [m] = %1.3f'%(self.a))
        print('Elongation k [-] = %1.3f'%(self.k))
        print('ni [per cubic m] = %1.3g'%(self.ni))
        print('Inductance L [Henry] = %1.3g'%(self.L))
        print('Btor [Tesla] = %1.3g'%(self.Btor))
        
        print("\n Initial Values and TQ:")
        
        print('Init. Plasma Current I0 [A] = %1.3g'%(self.I0))
        print('Init. RE Current Ir0 [A] = %1.3g'%(self.Ir0))
        print('Init. Temp. T0 [eV] = %1.3g'%(self.T0))
        print('Final Temp. Tf [eV] = %1.3g'%(self.Tf))
        print('Temp. decay time t0 [s] = %1.3g'%(self.t0))
        print('Init. Vloop V0 [V] = %1.3g'%(self.V0))
        print('Final Vloop Vf [V] = %1.3g'%(self.Vf))
        
        print("\n Impurity Settings:")
        
        print('Added D density [per cubic m] = %1.3g'%(self.nD))
        print('Added Impurity density [per cubic m] = %1.3g'%(self.n_imp))
        print('Added Impurity Z, A [-] = %1.0f, %1.0f'%(self.Z_imp,self.A_imp))
        
        print("\n RE Source Settings: \n")
        print('Primary: iDreicer = %d, iTritium = %d, iCompton = %d'%(self.iDreicer,self.iTritium,self.iCompton))
        print('Secondary: iAvalanche = %d'%(self.iAvalanche))
        
        
        print("\n Other Settings: \n")
        print('tstart [s] = %1.3g'%(self.tstart))
        print('tmax [s] = %1.3g'%(self.tmax))
        print('Zeff_scale [-] =' + str(self.Zeff_scale))
        print('device [-] = ' + str(self.device))


    def getTemp(self, t):
        return np.where(
            t < self.tstart,
            self.T0,
            (self.T0 - self.Tf) * np.exp(-(t - self.tstart) / self.t0) + self.Tf)

    def get_nimp(self, t):
        if self.prescibe_nz_evolution:
            n0, n1, n2 = 3.2e-4, 9.6e-4, 5.7e-4
            out = n0 + (n1-n0) * (1 + np.tanh((t-45e-3)/self.t0))/2 + (n2 - n1) * (1 + np.tanh((t-220e-3)/(3*self.t0)))/2
            return out*self.ni
        else:
            return self.n_imp * (t + 1e-6) / (t + 1e-6)

    def getImpurityIonization(self,T,n_imp):

        return getZbarTF(self.Z_imp, self.A_imp,
                    SI2cgs(self.A_imp*n_imp*const.m_p,'Mass Density'),
                    T)[0] # caluclate ionization level using LTE Thomas-Fermi model
        #return getNeZ(T) # Using ADAS Coronal Eqm. Model

    def getElecDensity(self,Z,n_imp=None):

        if n_imp is None:
            n_imp = self.n_imp
        else:
            n_imp = n_imp

        nfree = 1 * self.ni + 1 * self.nD + Z * n_imp
        ntot = self.ni + self.nD + self.Z_imp * n_imp
        return nfree, ntot

    def getPrimaryRates(self, E_star, ne, ntot, lnA, Zeff, T, Z, Epar, Ed):

        if self.iDreicer == 1:
            SD = getDreicerSource(E_star,ne,lnA,Zeff,T) # per m3 per s, Classical Dreicer
        elif self.iDreicer == 2:
            SD = getDreicerRateHesslow(self.ni,self.n_imp,self.Z_imp,Z,Epar/Ed,T)
        else:
            SD = 0
        
        if (self.iTritium == 1):
            ST = getTritBetaDecay(0.5*self.ni,E_star)  # per m3 per s
        else:
            ST = 0;

        if (self.iCompton == 1):
            if T >= 2.6e3:
                SC = getComptonSource(E_star,ntot,self.device+'-P') # Use Prompt Flux
            else:
                SC =  getComptonSource(E_star,ntot,self.device+'-NP') # Use Non Prompt Flux
        else:
            SC = 0;


        # Total Primary Contribution
        S =  SD + ST + SC # per m3 per s
        return S, SD, ST, SC
    
    def getAvalancheRates(self,E_star,ntot,ne,T,lnA,Zeff,Zion):
        # iAvalanche = 1 Rosenbluth-Putvinski * ntot/ne, 2 Simple RP, 3 = Hesslow, 0 = off
        # Avalanche
        if (self.iAvalanche == 0):
            Yav = 0;
        elif (self.iAvalanche == 1):
             Yav = (ntot/ne) * get_avalance_growth_rate_RP2(E_star,ne,0/self.R0,lnA,Zeff) # per second
        elif (self.iAvalanche == 2):
            Yav = 1 * get_avalance_growth_rate_RP2(E_star,ne,0/self.R0,lnA,Zeff) # per second
        elif (self.iAvalanche == 3): # Implement getHesslowAvalanchingRate 
         # Define arrays for species info, first element deuterium, second Neon (impurity)
            Z = np.array([1, self.Z_imp])             # Max ionization of D and Ne
            nj = np.array([self.ni, self.n_imp])          # Densities [m^-3]
            # interpolation 
            Zlow = int(np.floor(Zion))
            Zhigh = int(np.ceil(Zion))
            weight = Zion - Zlow # interpolation weight

            # Creating 2 separate arrays fo the low and high value
            Z0_low = np.array([1, Zlow])
            Z0_high = np.array([1, Zhigh])

            if (self.Z_imp==18):
                element = 'Ar'
            elif (self.Z_imp==10):
                element = 'Ne'
            elif (self.Z_imp==4):
                element = 'Be'
            elif (self.Z_imp==6):
                element = 'C'
            elif (self.Z_imp==74):
                element = 'W'
            else:
                element = None

            self.ln_aBar, self.ln_I = get_screening_constants(self.Z_imp, element)

            Y_low = getHesslowAvalanchingRate(E_star, Z, Z0_low, nj, T, self.ln_aBar, self.ln_I, B=self.Btor)
            Y_high = getHesslowAvalanchingRate(E_star, Z, Z0_high, nj, T, self.ln_aBar, self.ln_I, B=self.Btor)
            Yav = (1 - weight) * Y_low + weight * Y_high # set has the weighted average of the two
        return Yav

    def getVloop(self,T,t):
        return (self.V0 +
                (self.Vf-self.V0) * (1 + np.tanh((t-120e-3)/self.t0))/2
                - 2.0 * (1+np.tanh((t-170e-3)/self.t0))/2)
        
    def dYdt(self,t,Y):
        # Y[0] = Plasma Current [A]
        # Y[1] = RE current [A]
        # Y[2] = RE Kinetic Energy [J]
        # primary_sources = [iDreicer, iTritium, iCompton], 0 = off
        # iDreicer = 1 Classical; 2 = Hesslow NN
        # iAvalanche = 1 Rosenbluth-Putvinski * ntot/ne, 2 Simple RP, 3 = Hesslow, 0 = off

        Ip, Ir, Wkin = Y


        # Prescribed Thermal Quench
        T = self.getTemp(t)
        nimp = self.get_nimp(t) # impurity density
        Z = self.getImpurityIonization(T,nimp)
        ne, ntot = self.getElecDensity(Z,n_imp=nimp)
        Zeff = (self.ni + self.nD + nimp * Z**2) / ne * self.Zeff_scale

        lnA = getCoulombLog(ne,T) # Coulomb logarithm
        eta = 0.51 * getTransSpitzerResistivity(lnA,T,Zeff) * 0.01 # Ohm-m
        Epar = eta*(Ip - Ir) / (np.pi * self.a**2 * self.k) # Parallel E-field [N/C]
        Ec = getConnorHastieField(ne,lnA) # Connor-Hastie Critical Field, [N/C]
        Ed = getDreicerField(ne,lnA,T) # [N/C] Dreicer Field
        E_star = Epar / Ec # Normalized field [-]

        # Externally driven loop voltage
        Vloop = self.getVloop(T,t)


        # Primary Sources

        Sp, _, _, _ = self.getPrimaryRates(E_star, ne, ntot, lnA, Zeff, T, Z, Epar, Ed)
        j_primary = Sp * const.c * const.e # A/m**2/s
        
        # Avalanche Rate
        Yav = self.getAvalancheRates(E_star,ntot,ne,T,lnA,Zeff,Z)

        # Derivatives
        dIpdt = -2 * np.pi * self.R0 * Epar / self.L + Vloop / self.L # [A/s]
        dIrdt = j_primary * np.pi * self.a**2 + Yav * Ir # [A/s]
        if (E_star > 1):
            dWkindt = 2 * np.pi * self.R0 * Ir * Epar
        else:
            dWkindt = 0

        return [dIpdt, dIrdt, dWkindt]
    

    def solve(self):
        from scipy.integrate import solve_ivp
        t = np.linspace(0,self.tmax,int(1e4)) # times to solve at [s]
        sol = solve_ivp(self.dYdt, t_span=(0,max(t)),
                y0 = [self.I0,self.Ir0,0],
                t_eval=t,
                rtol=1e-8, atol=1e-10) # Solve
        Ip = sol.y[0]
        Ir = sol.y[1]
        Wk = sol.y[2]

        # Diagnostics
        if (self.idiag == True): # Calculate detailed output
            T = self.getTemp(t)
            Vloop = self.getVloop(T,t) # Volts
            nimp = self.get_nimp(t) # impurity density
            Z = self.getImpurityIonization(T,nimp)
            ne, ntot = self.getElecDensity(Z,nimp)
            Zeff = (self.ni + self.nD + nimp * Z**2) / ne * self.Zeff_scale
            lnA = getCoulombLog(ne,T) # Coulomb logarithm
            eta = 0.51 * getTransSpitzerResistivity(lnA,T,Zeff) * 0.01 # Ohm-m
            Epar = eta*(Ip - Ir) / (np.pi * self.a**2 * self.k) # Parallel E-field [N/C]
            Ec = getConnorHastieField(ne,lnA) # Connor-Hastie Critical Field, [N/C]
            Ed = getDreicerField(ne,lnA,T) # [N/C] Dreicer Field
            E_star = Epar / Ec # Normalized field [-]


            temp = self.iCompton

            self.iCompton = 0
            Sp, SD, ST, SC = self.getPrimaryRates(E_star, ne, ntot, lnA, Zeff, T, Z, Epar, Ed)

            self.iCompton = temp
            SC = np.zeros_like(E_star)
            if (self.iCompton == 1):
                for i in range(len(t)):
                    mode = self.device+'-P' if T[i] >= 2600 else self.device+'-NP'
                    SC[i] = getComptonSource(E_star[i], ntot[i], mode)

            if (self.iAvalanche != 3):
                Yav = self.getAvalancheRates(E_star,ntot,ne,T,lnA,Zeff,Z)
            else:
                Yav = np.zeros_like(t)
                for ii in range(Yav.shape[0]):
                    Yav[ii] = self.getAvalancheRates(E_star[ii],
                                                     ntot[ii],ne[ii],
                                                     T[ii],lnA[ii],
                                                     Zeff[ii],Z[ii])
            if (self.iTritium == 0):
                ST = np.zeros_like(E_star)
            if (self.iDreicer == 0):
                SD = np.zeros_like(E_star)


            self.diag = {
                't': t,
                'Te': T,
                'Vloop': Vloop,
                'ne': ne,
                'Zeff': Zeff,
                'nimp': nimp,
                'Epar': Epar,
                'E_star': E_star,
                'EoED': Epar/Ed,
                'eta': eta,
                'gamma_ava': Yav,
                'SD': SD,
                'ST': ST,
                'SC': SC
            }


        return t, Ip, Ir, Wk
        
        
        


