import importlib
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

# Add local path for modules
RE_path = "./"
sys.path.insert(1, RE_path)

import RE
from RE import *
import RESolver as solver

importlib.reload(RE)
importlib.reload(solver)


# ==========================================
# Example Solution using SPARC-like Parameters
# ==========================================


def getj0(I0, nx):
    r = np.linspace(0, 1, nx)
    jp0 = 13.3e6 * (1 - r**2) ** 0.5  # A/m**2
    javg = 2 * np.pi * np.trapz(jp0 * r, x=r) / (np.pi * 1**2)  # average jinit
    rmax = np.sqrt(
        I0 / (javg * np.pi)
    )  # radius required to preserve total current
    return jp0, rmax


# Constants
R0 = 1.85  # [m]
a = 0.57  # [m]
Zeff = 3
T = 10  # eV
ne = 3e20  # per m**3
#ne = 3e21;  # per m**3
#ne = 3e22;
#ne = 3e23; 
lnA = 15
eta = 0.51 * getTransSpitzerResistivity(lnA, T, Zeff) * 0.01  # Ohm-m

# Initial Conditions
nx 	= 40;
jp0,rmax= getj0(8.7e6,nx)
jr0 	= np.zeros_like(jp0)
r 	= np.linspace(1e-3,rmax,len(jp0)); # radial array [m]

constants = [rmax, R0, ne, T, Zeff, lnA, eta]
init_conditions = [jp0, jr0]


# ==========================================
# Configure and run the solver
# ==========================================

tmax		= 10e-3; # s
CFL		= 0.4;
#RE_sources=[1, 1, 1, 1],  # [iDreicer, iTritBeta, iCompton, iAvalanche]
iDreicer	= 0;
iTritBeta 	= 0; 
iCompton 	= 0;
iAvalanche 	= 1;
cre		= 0.9 * 3e8;
nprint		= 500;
verbose 	= False; #True;

#iVolumetric 	= ;
Y_gamma 	= 2e-5*5e19*np.exp(-5*(r/rmax)**2); # ph/m^3/s, gaussian profile estimate, exp(-5) < 1%, BR ~ 2e-5, Pfus ~ 5e19 n/s
#Y_gamma*=1e5;
E_gamma 	= 16.7; # MeV, DT gamma

importlib.reload(solver)

iVolumetric 	= 0;
vol_gamma_data 	= [iVolumetric,Y_gamma,E_gamma];
out0 = solver.solve(
    constants=constants,
    init_conditions=init_conditions,
    tmax=tmax,
    CFL=CFL,
    RE_sources=[iDreicer, iTritBeta, iCompton, iAvalanche],
    cre=cre,
    nprint=nprint,
    verbose=verbose,
    vol_gamma_data=[iVolumetric,Y_gamma,E_gamma],
)

iVolumetric 	= 1;
out1 = solver.solve(
    constants=constants,
    init_conditions=init_conditions,
    tmax=tmax,
    CFL=CFL,
    RE_sources=[iDreicer, iTritBeta, iCompton, iAvalanche],
    cre=cre,
    nprint=nprint,
    verbose=verbose,
    vol_gamma_data=[iVolumetric,Y_gamma,E_gamma],
)

# ==========================================
# Show results / Plotting
# ==========================================

def plot_Ip_Ir(out,clr,lbl):
	plt.plot(out['t'],out['Ip'],ls='-',color=clr,label=lbl+', Ip');
	plt.plot(out['t'],out['Ir'],ls='--',color=clr,label=lbl+', Ir');
	
def plot_jp_jr(out,clr,lbl):
	plt.plot(out['r']/out['rmax'],out['jp'],ls='-',color=clr,label=lbl+', jp');
	plt.plot(out['r']/out['rmax'],out['jr'],ls='--',color=clr,label=lbl+', jr');

# Ip, Ir
fig = plt.figure();
plot_Ip_Ir(out0,'k','no Vol');
plot_Ip_Ir(out1,'r','Volumetric effects');
plt.xlabel('t(s)');
plt.ylabel('(A)');
plt.grid();
plt.legend();
fig.tight_layout();
fig.show();

# jp, jr
fig = plt.figure();
plot_jp_jr(out0,'k','no Vol');
plot_jp_jr(out1,'r','Volumetric effects');
plt.xlabel('r/a');
plt.ylabel('(A/m^2)');
plt.grid();
plt.legend();
fig.tight_layout();
fig.show();


# Add custom visualization code or display `out` here if needed.
print("Simulation complete.")
