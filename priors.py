import numpy as np
import astropy
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as const

def uniform_pdf(x, a, b):
    return np.where(np.logical_and(x >= a, x <= b), 1 / (b - a), 0)

def uniform_in_cosine_pdf(x, a = -np.pi/2, b = np.pi/2):
    return np.where(np.logical_and(x >= a, x <= b), 0.5 * np.cos(x), 0)

def uniform_in_sine_pdf(x, a = 0, b = np.pi):
    return np.where(np.logical_and(x >= a, x <= b), 0.5 * np.sin(x), 0)

def uniform_in_differential_comoving_volume_pdf(x, a, b, z):
    #norm_factor = norm_factor_distance_prior(a, b)
    return np.where(np.logical_and(z >= a, z <= b), 
                cosmo.differential_comoving_volume(z).value * (x / (1 + z) + (const.c.value / 1000) * (1 + z) / (cosmo.H(0).value * cosmo.efunc(z)))**(-1) / (1 + z), 0)

def uniform_in_distance_squared_pdf(x, a = 10, b = 10000):
    return np.where(np.logical_and(x >= a, x <= b), x**2, 0)