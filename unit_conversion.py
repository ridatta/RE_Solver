import numpy as np
import scipy.constants as const

def cgs2SI(val, type):
    """
    Convert from cgs to SI units.
    
    Parameters:
    val (float or np.ndarray): The value(s) in CGS units to convert.
    type (str): The type of physical quantity to convert.

    Returns:
    np.ndarray: The value(s) in SI units.
    """

    # Define physical constants
    alph = 1e2  # [cm/m]
    beta = 1e7  # [erg/J]
    gam = 1e3   # [g/kg]
    ep0 = const.epsilon_0  # Vacuum permittivity [F/m]
    mu0 = const.mu_0  # Vacuum permeability [H/m]
    
    # Determine the conversion factor
    if type == 'Capacitance':
        k = alph * (4 * np.pi * ep0)**-1
    elif type == 'Charge':
        k = np.sqrt(alph * beta / (4 * np.pi * ep0))
    elif type == 'Charge Density':
        k = np.sqrt(beta / (4 * np.pi * alph**5 * ep0))
    elif type == 'Conductance':
        k = alph * (4 * np.pi * ep0)**-1
    elif type == 'Conductivity':
        k = (4 * np.pi * ep0)**-1
    elif type == 'Current':
        k = np.sqrt(alph * beta / (4 * np.pi * ep0))
    elif type == 'Current Density':
        k = (1 / alph**2) * np.sqrt(alph * beta / (4 * np.pi * ep0))
    elif type == 'Density':
        k = alph**-3
    elif type == 'Electric Field':
        k = np.sqrt(4 * np.pi * beta * ep0 / alph**3)
    elif type == 'EMF':
        k = alph * np.sqrt(4 * np.pi * beta * ep0 / alph**3)
    elif type == 'Electric Potential':
        k = np.sqrt(4 * np.pi * beta * ep0 / alph)
    elif type == 'Electric Conductivity':
        k = (4 * np.pi * ep0)**-1
    elif type == 'Energy':
        k = beta
    elif type == 'Energy Density':
        k = beta / alph**3
    elif type == 'Force':
        k = beta / alph
    elif type == 'Impedance':
        k = 4 * np.pi * ep0 / alph
    elif type == 'Inductance':
        k = 4 * np.pi * ep0 / alph
    elif type == 'Length':
        k = alph
    elif type == 'Magnetic Induction':
        k = np.sqrt(4 * np.pi * beta / (alph**3 * mu0))
    elif type == 'Magnetic Field':
        k = np.sqrt(4 * np.pi * beta / (alph**3 * mu0))
    elif type == 'Magnetic Flux':
        k = np.sqrt(4 * np.pi * beta / (alph**3 * mu0)) * alph**2
    elif type == 'Magnetic Intensity':
        k = np.sqrt(4 * np.pi * mu0 * beta / alph**3)
    elif type == 'Magnetic Moment':
        k = 1e3
    elif type == 'Magnetization':
        k = np.sqrt(4 * np.pi * mu0 * beta / alph**3)
    elif type == 'Magnetomotance':
        k = alph * np.sqrt(4 * np.pi * mu0 * beta / alph**3)
    elif type == 'Mass':
        k = beta / alph**2
    elif type == 'Mass Density':
        k = gam * alph**-3
    elif type == 'Momentum':
        k = beta / alph
    elif type == 'Momentum Density':
        k = beta / (alph**4)
    elif type == 'Polarization':
        k = np.sqrt(beta / (4 * np.pi * alph**5 * ep0))
    elif type == 'Power':
        k = beta
    elif type == 'Power Density':
        k = beta / alph**3
    elif type == 'Pressure':
        k = beta / alph**3
    elif type == 'Reluctance':
        k = 4 * np.pi * 1e-9
    elif type == 'Resistance':
        k = 4 * np.pi * ep0 / alph
    elif type == 'Resistivity':
        k = 4 * np.pi * ep0 / alph * alph
    elif type == 'Velocity':
        k = alph
    else:
        raise ValueError("Type not supported")

    return val / k




def SI2cgs(val, type):
    """
    Convert from SI to cgs units.
    
    Parameters:
        val: The value to be converted.
        type: The type of physical quantity.
    
    Returns:
        The value in cgs units.
    """
    # Constants
    alph = 10**2  # [cm/m]
    beta = 1e7    # [erg/J]
    gam = 1e3     # [g/kg]
    
    ep0 = const.epsilon_0  # Vacuum permittivity [F/m]
    mu0 = const.mu_0  # Vacuum permeability [H/m]
    
    # Define the conversion factor based on the type
    k = None
    if type == 'Capacitance':
        k = alph * (4 * np.pi * ep0)**-1
    elif type == 'Charge':
        k = np.sqrt(alph * beta / (4 * np.pi * ep0))
    elif type == 'Charge Density':
        k = np.sqrt(beta / (4 * np.pi * alph**5 * ep0))
    elif type == 'Conductance':
        k = alph * (4 * np.pi * ep0)**-1
    elif type == 'Conductivity':
        k = (4 * np.pi * ep0)**-1
    elif type == 'Current':
        k = np.sqrt(alph * beta / (4 * np.pi * ep0))
    elif type == 'Current Density':
        k = 1 / alph**2 * np.sqrt(alph * beta / (4 * np.pi * ep0))
    elif type == 'Density':
        k = alph**-3
    elif type == 'Electric Field':
        k = np.sqrt(4 * np.pi * beta * ep0 / alph**3)
    elif type == 'EMF':
        k = alph * np.sqrt(4 * np.pi * beta * ep0 / alph**3)
    elif type == 'Electric Potential':
        k = np.sqrt(4 * np.pi * beta * ep0 / alph)
    elif type == 'Electric Conductivity':
        k = (4 * np.pi * ep0)**-1
    elif type == 'Energy':
        k = beta
    elif type == 'Energy Density':
        k = beta / alph**3
    elif type == 'Force':
        k = beta / alph
    elif type == 'Impedance':
        k = 4 * np.pi * ep0 / alph
    elif type == 'Inductance':
        k = 4 * np.pi * ep0 / alph
    elif type == 'Length':
        k = alph
    elif type == 'Magnetic Induction':
        k = np.sqrt(4 * np.pi * beta / (alph**3 * mu0))
    elif type == 'Magnetic Field':
        k = np.sqrt(4 * np.pi * beta / (alph**3 * mu0))
    elif type == 'Magnetic Flux':
        k = np.sqrt(4 * np.pi * beta / (alph**3 * mu0)) * alph**2
    elif type == 'Magnetic Intensity':
        k = np.sqrt(4 * np.pi * mu0 * beta / alph**3)
    elif type == 'Magnetic Moment':
        k = 1e3
    elif type == 'Magnetization':
        k = np.sqrt(4 * np.pi * mu0 * beta / alph**3)
    elif type == 'Magnetomotance':
        k = alph * np.sqrt(4 * np.pi * mu0 * beta / alph**3)
    elif type == 'Mass':
        k = beta / alph**2
    elif type == 'Mass Density':
        k = gam * alph**-3
    elif type == 'Momentum':
        k = beta / alph
    elif type == 'Momentum Density':
        k = beta / (alph**4)
    elif type == 'Polarization':
        k = np.sqrt(beta / (4 * np.pi * alph**5 * ep0))
    elif type == 'Power':
        k = beta
    elif type == 'Power Density':
        k = beta / alph**3
    elif type == 'Pressure':
        k = beta / alph**3
    elif type == 'Reluctance':
        k = 4 * np.pi * 1e-9
    elif type == 'Resistance':
        k = 4 * np.pi * ep0 / alph
    elif type == 'Resistivity':
        k = 4 * np.pi * ep0 / alph * alph
    elif type == 'Velocity':
        k = alph
    else:
        raise ValueError('Type not supported')
    
    # Convert the value
    out = val * k
    return out

