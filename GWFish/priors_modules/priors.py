#    Copyright (c) 2024 Ulyana Dupletsa <ulyana.dupletsa@gssi.it>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.


import numpy as np
import astropy
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as const

def uniform_pdf(x, a, b):
    """
    Calculate the probability density function (PDF) of a uniform distribution.

    Parameters:
    x (float or array-like): The value(s) at which to evaluate the PDF.
    a (float): The lower bound of the uniform distribution.
    b (float): The upper bound of the uniform distribution.

    Returns:
    float or array-like: The PDF value(s) at the given x.

    """
    return np.where(np.logical_and(x >= a, x <= b), 1 / (b - a), 0)

def cosine_pdf(x, a = -np.pi/2, b = np.pi/2):
    """
    Calculate the probability density function (PDF) of a distribution uniform
    in the sine of x.

    Parameters:
    x (float or array-like): The input value(s) at which to evaluate the PDF in rad.
    a (float, optional): The lower bound of the uniform distribution. Defaults to -np.pi/2.
    b (float, optional): The upper bound of the uniform distribution. Defaults to np.pi/2.

    Returns:
    float or array-like: The value(s) of the PDF at the given input(s).
    """
    return np.where(np.logical_and(x >= a, x <= b), 0.5 * np.cos(x), 0)

def sine_pdf(x, a = 0, b = np.pi):
    """
    Calculate the probability density function (PDF) of a distribution uniform 
    in the cosine of x.

    Parameters:
    x (float or array-like): The input value(s) at which to evaluate the PDF in rad.
    a (float, optional): The lower bound of the uniform distribution. Defaults to 0.
    b (float, optional): The upper bound of the uniform distribution. Defaults to np.pi.

    Returns:
    float or array-like: The value(s) of the PDF at the given input(s).
    """
    return np.where(np.logical_and(x >= a, x <= b), 0.5 * np.sin(x), 0)

def uniform_in_source_frame_pdf(x, a = 0.0001, b = 5):
    """
    Calculate the probability density function (PDF) of a uniform distribution 
    in the differential comoving volume: 
    p(dL) propto dV/dz * (ddL/dz)^(-1) / (1 + z)
    where dL is the luminosity distance, given by dL = c/H0 * (1 + z) * int_0^z dz'/E(z')

    Parameters:
    x (float or array-like): Luminosity distance value(s) in Mpc at which to evaluate the PDF.
    a (float): The lower bound of the uniform distribution in redshift. Defaults to 0.0001.
    b (float): The upper bound of the uniform distribution in redshift. Defaults to 5.
    

    Returns:
    float or array-like: The PDF value(s) at the given x.

    """
    # find the redshifts corresponding to the input luminosity distance values
    zz = np.linspace(0.0001, 5, 1000)
    dd = cosmo.luminosity_distance(zz).value
    z = np.interp(x, dd, zz)
    # norm_factor (so that probability sum up to 1) neglected
    # remember to divide const.c by 1000 to convert to km/s
    return np.where(np.logical_and(z >= a, z <= b), 
                cosmo.differential_comoving_volume(z).value * (x / (1 + z) + (const.c.value / 1000) * (1 + z) / (cosmo.H(0).value * cosmo.efunc(z)))**(-1) / (1 + z), 0)

def uniform_in_comoving_volume_pdf(x, a = 0.0001, b = 5):
    """
    Calculate the probability density function (PDF) of a uniform distribution 
    in the differential comoving volume: 
    p(dL) propto dV/dz * (ddL/dz)^(-1)
    where dL is the luminosity distance, given by dL = c/H0 * (1 + z) * int_0^z dz'/E(z')

    Parameters:
    x (float or array-like): Luminosity distance value(s) in Mpc at which to evaluate the PDF.
    a (float): The lower bound of the uniform distribution in redshift. Defaults to 0.0001.
    b (float): The upper bound of the uniform distribution in redshift. Defaults to 5.

    Returns:
    float or array-like: The PDF value(s) at the given x.

    """
    # find the redshifts corresponding to the input luminosity distance values
    zz = np.linspace(0.0001, 5, 1000)
    dd = cosmo.luminosity_distance(zz).value
    z = np.interp(x, dd, zz)
    # norm_factor (so that probability sum up to 1) neglected
    # remember to divide const.c by 1000 to convert to km/s
    return np.where(np.logical_and(z >= a, z <= b), 
                cosmo.differential_comoving_volume(z).value * (x / (1 + z) + (const.c.value / 1000) * (1 + z) / (cosmo.H(0).value * cosmo.efunc(z)))**(-1), 0)


def uniform_in_distance_squared_pdf(x, a = 10, b = 10000):
    """
    Calculate the probability density function (PDF) of a uniform distribution
    in the distance squared: p(dL) propto dL^2

    Parameters:
    x (float or array-like): The value(s) at which to evaluate the PDF in Mpc.
    a (float, optional): The lower bound of the uniform distribution. Defaults to 10 Mpc.
    b (float, optional): The upper bound of the uniform distribution. Defaults to 10000 Mpc.

    Returns:
    float or array-like: The PDF value(s) at the given x.

    """
    # norm_factor (so that probability sum up to 1) neglected
    return np.where(np.logical_and(x >= a, x <= b), x**2, 0)