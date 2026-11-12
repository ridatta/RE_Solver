# Helper Functions

import os
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

# Radiation Model Taken from https://github.com/ukaea/PROCESS.git

LZ_DIR = "/Users/rishabhdatta/MIT Dropbox/Rishabh Datta/Rishabh_EP_Group/Code/Radiation_ADAS/lz_non_corona_14_elements"

_SUFFIX = {"lz": "_lz_tau.dat", "z": "_z_tau.dat", "z2": "_z2_tau.dat"}

@lru_cache(maxsize=None)

def _table(element, quantity, data_dir):
    fname = f"{element.capitalize():_<2}{_SUFFIX[quantity]}"
    Te, blocks = load_lz(os.path.join(data_dir, fname))
    return np.log10(Te), blocks[np.inf]          # coronal-equilibrium block


def z_coronal(T, element, units="eV", kind="mean", data_dir=LZ_DIR):
    """Coronal-equilibrium impurity charge.

    kind = 'mean' -> <Z>        (use for electron contribution: n_e += <Z> n_Z)
           'rms'  -> sqrt(<Z^2>)
           'z2'   -> <Z^2>      (use for Z_eff: sum n_Z <Z^2> / n_e)
    """
    scale = {"ev": 1.0, "kev": 1e3}[units.lower()]
    logT = np.log10(np.asarray(T, dtype=float) * scale)
    logTe, z = _table(element, "z" if kind == "mean" else "z2", data_dir)
    out = np.interp(logT, logTe, z, left=np.nan, right=np.nan)
    if kind == "z2":
        out = out**2
    return out.item() if out.ndim == 0 else out


def load_lz(path):
    """Parse a PROCESS/ADAS *_lz_tau.dat file -> (Te_eV, {tau_ms or np.inf: Lz})."""
    blocks, key, vals = {}, None, []

    def flush():
        if key is not None:
            blocks[key] = np.array(vals, dtype=float)

    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("C"):
                break
            if s.startswith("Te[eV]"):
                flush(); key, vals = "Te", []
            elif "tau[ms]=" in s:
                flush(); key, vals = float(s.split("=")[-1]), []
            elif "equilibrium" in s:
                flush(); key, vals = np.inf, []
            elif key is not None and s:
                vals.extend(s.split())
    flush()
    return blocks.pop("Te"), blocks


@lru_cache(maxsize=None)
def _coronal_table(element, data_dir):
    """Load (log10 Te, log10 Lz) for the coronal-equilibrium curve, cached per element."""
    fname = f"{element.capitalize():_<2}_lz_tau.dat"     # 'C' -> 'C__lz_tau.dat'
    path = os.path.join(data_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No L_z file for element '{element}': {path}")
    Te, blocks = load_lz(path)
    return np.log10(Te), np.log10(blocks[np.inf])


def lz_coronal(T, element, units="eV", data_dir=LZ_DIR):
    """Coronal-equilibrium radiative loss function L_z [W m^3].
    # Radiative power is given by ne * n_ion * Lz

    Parameters
    ----------
    T        : float or array, electron temperature
    element  : str, e.g. 'Ar', 'W', 'C', 'Ne'
    units    : 'eV' (default) or 'keV'
    data_dir : folder containing the *_lz_tau.dat files

    Interpolates linearly in log10(T)-log10(Lz). Returns NaN outside the
    tabulated range (0.25 eV - 500 keV). Data are for n_e = 1e20 m^-3.
    """
    scale = {"ev": 1.0, "kev": 1e3}[units.lower()]
    T_eV = np.asarray(T, dtype=float) * scale
    logTe, logLz = _coronal_table(element, data_dir)
    out = 10 ** np.interp(np.log10(T_eV), logTe, logLz, left=np.nan, right=np.nan)
    return out.item() if out.ndim == 0 else out

def get_Prad(T,ne,ni,elm):
    # T [eV]
    # ne = free electron density [per cubic m]
    # ni = ion density [per cubic m]
    # elm = ["H", "He", "Be", "C", "N", "O", "Ne", "Si", "Ar", "Fe", "Ni", "Kr", "Xe", "W"]
    Lz = lz_coronal(T,elm) # W.m3
    return ni * ne * Lz # W per cubic m


if __name__ == "__main__":
    print(lz_coronal(1000, "Ar"))                    # 2.74e-33 W m^3 at 1 keV
    print(lz_coronal([0.01, 1, 10], "W", units="keV"))
    
