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
Zeff = 1
T = 10  # eV
ne = 3e20  # per m**3
lnA = 15
eta = 0.51 * getTransSpitzerResistivity(lnA, T, Zeff) * 0.01  # Ohm-m

# Initial Conditions
jp0, rmax = getj0(8.7e6, 128)
jr0 = np.zeros_like(jp0)

constants = [rmax, R0, ne, T, Zeff, lnA, eta]
init_conditions = [jp0, jr0]


# ==========================================
# Configure and run the solver
# ==========================================

importlib.reload(solver)

out = solver.solve(
    constants=constants,
    init_conditions=init_conditions,
    tmax=10e-3,
    CFL=0.4,
    RE_sources=[1, 1, 1, 1],  # [iDreicer, iTritBeta, iCompton, iAvalanche]
    cre=0.1 * 3e8,
    nprint=500,
    verbose=True,
)


# ==========================================
# Show results / Plotting
# ==========================================

# Add custom visualization code or display `out` here if needed.
print("Simulation complete.")
