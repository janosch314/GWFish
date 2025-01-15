
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.


import numpy as np
import pandas as pd
import astropy
from astropy.cosmology import Planck18 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as const

from GWFish.modules.minimax_tilting_sampler import *

def uniform(x, a, b):
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

def uniform_in_cosine(x, a = -np.pi/2, b = np.pi/2):
    """
    Calculate the probability of a distribution uniform
    in the sine of x.

    !!! This is NOT normalized !!!

    Parameters:
    x (float or array-like): The input value(s) at which to evaluate the PDF in rad.
    a (float, optional): The lower bound of the uniform distribution. Defaults to -np.pi/2.
    b (float, optional): The upper bound of the uniform distribution. Defaults to np.pi/2.

    Returns:
    float or array-like: The value(s) of the prior at the given input(s).
    """
    return np.where(np.logical_and(x >= a, x <= b), np.cos(x), 0)

def uniform_in_sine(x, a = 0, b = np.pi):
    """
    Calculate the probability of a distribution uniform 
    in the cosine of x.

    !!! This is NOT normalized !!!

    Parameters:
    x (float or array-like): The input value(s) at which to evaluate the PDF in rad.
    a (float, optional): The lower bound of the uniform distribution. Defaults to 0.
    b (float, optional): The upper bound of the uniform distribution. Defaults to np.pi.

    Returns:
    float or array-like: The value(s) of the prior at the given input(s).
    """
    return np.where(np.logical_and(x >= a, x <= b), np.sin(x), 0)

def uniform_in_comoving_volume_and_source_frame(x, a = 10, b = 10000):
    """
    Calculate the probability of a uniform distribution 
    in the differential comoving volume: 
    p(dL) propto dV/dz * (ddL/dz)^(-1) / (1 + z)
    where dL is the luminosity distance, given by dL = c/H0 * (1 + z) * int_0^z dz'/E(z')

    !!! This is NOT normalized !!!

    Parameters:
    x (float or array-like): Luminosity distance value(s) in Mpc at which to evaluate the PDF.
    a (float): The lower bound of the uniform distribution in redshift. Defaults to 0.0001.
    b (float): The upper bound of the uniform distribution in redshift. Defaults to 5.
    
    Returns:
    float or array-like: The prior value(s) at the given x.

    """
    # find the redshifts corresponding to the input luminosity distance values
    # convert distance values a nd b to redshift
    aa = z_at_value(cosmo.luminosity_distance, a * u.Mpc)
    bb = z_at_value(cosmo.luminosity_distance, b * u.Mpc)

    zz = np.linspace(aa, bb, 1000)
    dd = cosmo.luminosity_distance(zz).value
    z = np.interp(x, dd, zz)

    # remember to divide const.c by 1000 to convert to km/s
    return np.where(np.logical_and(z >= aa, z <= bb), 
            cosmo.differential_comoving_volume(z).value * (x / (1 + z) + (const.c.value / 1000) * (1 + z) / (cosmo.H(0).value * cosmo.efunc(z)))**(-1) / (1 + z), 0)

def uniform_in_comoving_volume(x, a = 10, b = 10000):
    """
    Calculate the probability of a uniform distribution 
    in the differential comoving volume: 
    p(dL) propto dV/dz * (ddL/dz)^(-1)
    where dL is the luminosity distance, given by dL = c/H0 * (1 + z) * int_0^z dz'/E(z')

    !!! This is NOT normalized !!!

    Parameters:
    x (float or array-like): Luminosity distance value(s) in Mpc at which to evaluate the PDF.
    a (float): The lower bound of the uniform distribution in redshift. Defaults to 0.0001.
    b (float): The upper bound of the uniform distribution in redshift. Defaults to 5.

    Returns:
    float or array-like: The prior value(s) at the given x.

    """
    # find the redshifts corresponding to the input luminosity distance values
    # convert distance values a nd b to redshift
    aa = z_at_value(cosmo.luminosity_distance, a * u.Mpc)
    bb = z_at_value(cosmo.luminosity_distance, b * u.Mpc)

    # interpolate the redshifts
    zz = np.linspace(aa, bb, 1000)
    dd = cosmo.luminosity_distance(zz).value
    z = np.interp(x, dd, zz)

    # remember to divide const.c by 1000 to convert to km/s
    return np.where(np.logical_and(z >= aa, z <= bb), 
            cosmo.differential_comoving_volume(z).value * (x / (1 + z) + (const.c.value / 1000) * (1 + z) / (cosmo.H(0).value * cosmo.efunc(z)))**(-1), 0)

def uniform_in_distance_squared(x, a = 10, b = 10000):
    """
    Calculate the probability of a uniform distribution
    in the distance squared: p(dL) propto dL^2

    !!! This is NOT normalized !!!

    Parameters:
    x (float or array-like): The value(s) at which to evaluate the PDF in Mpc.
    a (float, optional): The lower bound of the uniform distribution. Defaults to 10 Mpc.
    b (float, optional): The upper bound of the uniform distribution. Defaults to 10000 Mpc.
    norm (bool, optional): Whether to normalize the PDF. Defaults

    Returns:
    float or array-like: The prior value(s) at the given x.

    """
    return np.where(np.logical_and(x >= a, x <= b), x**2, 0)

def uniform_in_component_masses_chirp_mass(mChirp_samples, mChirp_min, mChirp_max):
    """
    Calculate the probability of the joint prior on the chirp mass and mass ratio
    which is uniform in the component masses

    !!! This is NOT normalized !!!

    Parameters:
    mChirp_samples (array-like): The chirp mass value(s) at which to evaluate the PDF.
    mChirp_min (float): The lower bound of the uniform distribution in chirp mass.
    mChirp_max (float): The upper bound of the uniform distribution in chirp mass.

    Returns:
    float or array-like: The prior value(s) at the given x.

    """
    return np.where(np.logical_and(mChirp_samples >= mChirp_min, mChirp_samples <= mChirp_max), mChirp_samples, 0)

def uniform_in_component_masses_mass_ratio(q_samples, q_min, q_max):
    """
    Calculate the probability of the joint prior on the chirp mass and mass ratio
    which is uniform in the component masses

    !!! This is NOT normalized !!!

    Parameters:
    q_samples (array-like): The mass ratio value(s) at which to evaluate the PDF.
    q_min (float): The lower bound of the uniform distribution in mass ratio.
    q_max (float): The upper bound of the uniform distribution in mass ratio.

    Returns:
    float or array-like: The prior value(s) at the given x.

    """
    return np.where(np.logical_and(q_samples >= q_min, q_samples <= q_max), q_samples**(-6./5.) * (1 + q_samples)**(2./5.), 0)

def get_available_prior_functions():
    """
    Get the list of available prior functions.

    Returns:
    list: The list of available prior functions.

    """
    return ['uniform', 'uniform_in_cosine', 'uniform_in_sine', 'uniform_in_comoving_volume_and_source_frame', 'uniform_in_comoving_volume', 
            'uniform_in_distance_squared', 'uniform_in_component_masses_chirp_mass', 'uniform_in_component_masses_mass_ratio']

def get_default_priors_dict(params):
    """
    Get the dictionary of priors for the given parameters. 
    The format of the dictionary is 
    {\'param_name\': {\'prior_type\': \'prior_function\', \'lower_prior_bound\': lower_bound, \'upper_prior_bound\': upper_bound}}

    Parameters:
    params (list): The list of parameters.

    Returns:
    dict: The dictionary of priors.

    """
    priors_dict = {}
    for var in params:
        if var == 'mass_1':
            priors_dict[var] = {'prior_type': 'uniform', 'lower_prior_bound': 5, 'upper_prior_bound': 100}
        elif var == 'mass_2':
            priors_dict[var] = {'prior_type': 'uniform', 'lower_prior_bound': 5, 'upper_prior_bound': 100}
        elif var == 'mass_ratio':
            priors_dict[var] = {'prior_type': 'uniform_in_component_masses_mass_ratio', 'lower_prior_bound': 0., 'upper_prior_bound': 0.99}
        elif var == 'chirp_mass':
            priors_dict[var] = {'prior_type': 'uniform_in_component_masses_chirp_mass', 'lower_prior_bound': 5, 'upper_prior_bound': 100}
        elif var == 'luminosity_distance':
            priors_dict[var] = {'prior_type': 'uniform_in_comoving_volume_and_source_frame', 'lower_prior_bound': 10, 'upper_prior_bound': 10000}
        elif var == 'theta_jn':
            priors_dict[var] = {'prior_type': 'uniform_in_sine', 'lower_prior_bound': 0, 'upper_prior_bound': np.pi}
        elif var == 'ra':
            priors_dict[var] = {'prior_type': 'uniform', 'lower_prior_bound': 0, 'upper_prior_bound': 2*np.pi}
        elif var == 'dec':
            priors_dict[var] = {'prior_type': 'uniform_in_cosine', 'lower_prior_bound': -np.pi/2, 'upper_prior_bound': np.pi/2}
        elif var == 'psi':
            priors_dict[var] = {'prior_type': 'uniform', 'lower_prior_bound': 0, 'upper_prior_bound': np.pi}
        elif var == 'phase':
            priors_dict[var] = {'prior_type': 'uniform', 'lower_prior_bound': 0, 'upper_prior_bound': 2*np.pi}
        elif var == 'geocent_time':
            priors_dict[var] = {'prior_type': 'uniform', 'lower_prior_bound': 0, 'upper_prior_bound': 1e10}
        elif var == 'a_1':
            priors_dict[var] = {'prior_type': 'uniform', 'lower_prior_bound': 0, 'upper_prior_bound': 0.99}
        elif var == 'a_2':
            priors_dict[var] = {'prior_type': 'uniform', 'lower_prior_bound': 0, 'upper_prior_bound': 0.99}
        elif var == 'tilt_1':
            priors_dict[var] = {'prior_type': 'uniform_in_sine', 'lower_prior_bound': 0, 'upper_prior_bound': np.pi}
        elif var == 'tilt_2':
            priors_dict[var] = {'prior_type': 'uniform_in_sine', 'lower_prior_bound': 0, 'upper_prior_bound': np.pi}
        elif var == 'phi_12':
            priors_dict[var] = {'prior_type': 'uniform', 'lower_prior_bound': 0, 'upper_prior_bound': 2*np.pi}
        elif var == 'phi_jl':
            priors_dict[var] = {'prior_type': 'uniform', 'lower_prior_bound': 0, 'upper_prior_bound': 2*np.pi}
        elif var == 'lambda_1':
            priors_dict[var] = {'prior_type': 'uniform', 'lower_prior_bound': 0, 'upper_prior_bound': 5000}
        elif var == 'lambda_2':
            priors_dict[var] = {'prior_type': 'uniform', 'lower_prior_bound': 0, 'upper_prior_bound': 5000}
    return priors_dict

def print_default_priors_dict(params):
    """
    Print the default priors dictionary for the given parameters.

    Parameters:
    params (list): The list of parameters.

    """
    priors_dict = get_default_priors_dict(params)
    for var in params:
        print('{}: {} in the interval [{}, {}]'.format(var, priors_dict[var]['prior_type'], priors_dict[var]['lower_prior_bound'], priors_dict[var]['upper_prior_bound']))

    

def get_truncated_likelihood_samples(params, mean_values, cov_matrix, num_samples, min_array = None, max_array = None):
    """
    Generate truncated likelihood samples for the given parameters.

    Parameters:
    params (list): The list of parameters.
    mean_values (array-like): The mean values for the parameters.
    cov_matrix (array-like): The covariance matrix for the parameters.
    num_samples (int): The number of samples to generate.
    min_array (array-like): The lower bounds for the parameters.
    mx_array (array-like): The upper bounds for the parameters.

    ! The default prior ranges are used if the min_array and max_array are not provided.
    
    Returns:
    array-like: Dataframe of truncated likelihood samples for the params.

    """
    if min_array is None:
        priors_dict = get_default_priors_dict(params)
        min_array = np.array([priors_dict[key]['lower_prior_bound'] for key in params])
    if max_array is None:
        priors_dict = get_default_priors_dict(params)
        max_array = np.array([priors_dict[key]['upper_prior_bound'] for key in params])
    tmvn = TruncatedMVN(mean_values, cov_matrix, min_array, max_array)
    samples = tmvn.sample(num_samples)

    return pd.DataFrame(samples.T, columns=params)

def get_posteriors_samples(params, likelihood_samples, num_posterior_samples, priors_dict = None):
    """
    Calculate the posterior samples given the likelihood samples and the priors.

    Parameters:
    params (list): The list of parameters.
    likelihood_samples (array-like): Dataframe of likelihood samples for the params.
    num_posterior_samples (int): The number of posterior samples to generate.
    priors_dict (dict): The dictionary of prior: should contain for each param, the type of prior and the bounds; if None is passed, the default priors are used.
    
    Returns:
    array-like: Dataframe of posterior samples for params.

    """
    if priors_dict is None:
        priors_dict = get_default_priors_dict(params) 

    prior = np.ones(len(likelihood_samples))
    for var in params:
        prior *= globals()[priors_dict[var]['prior_type']](likelihood_samples[var].to_numpy(), priors_dict[var]['lower_prior_bound'], priors_dict[var]['upper_prior_bound'])

    prior /= np.sum(prior)
    idx = np.random.choice(len(likelihood_samples), size=num_posterior_samples, replace=True, p=prior)

    return likelihood_samples.iloc[idx]
