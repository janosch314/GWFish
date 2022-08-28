"""
This module is aimed at computing detection horizons.

The thing we want to compute is the luminosity distance at which a given
signal will be detected (with a given SNR).
"""

import warnings
import numpy as np
from tqdm import tqdm

from astropy.cosmology import Planck18
import astropy.cosmology as cosmology
import astropy.units as u

from scipy.optimize import fsolve

from .detection import SNR, Detector, projection
from .waveforms import hphc_amplitudes

DEFAULT_RNG = np.random.default_rng(seed=1)

WAVEFORM_MODEL = 'lalsim_IMRPhenomD'
MIN_REDSHIFT = 1e-8

def compute_SNR(params: dict, detector: Detector, waveform_model: str = WAVEFORM_MODEL):
    
    polarizations, timevector = hphc_amplitudes(
        waveform_model, 
        params,
        detector.frequencyvector,
        plot=None
    )
    
    signal = projection(
        params,
        detector,
        polarizations,
        timevector
    )
    
    component_SNRs = SNR(detector, signal)
    if np.all(component_SNRs==0.):
        raise ValueError('The SNR is zero in all components!')
    return np.sqrt(np.sum(component_SNRs**2))

def horizon(
    params: dict,
    detector: Detector,
    target_SNR: int = 9, 
    waveform_model: str = WAVEFORM_MODEL,
    cosmology_model: cosmology.Cosmology = Planck18
    ):
    """
    Given the parameters for a GW signal and a detector, this function 
    computes the luminosity distance and corresponding redshift 
    (as connected by a given cosmology) which the signal would 
    need to be at in order to have a given redshift - by default, 9.
    
    Returns:
    
    distance in Mpc, redshift
    """
    
    if 'redshift' in params or 'luminosity_distance' in params:
        warnings.warn('The redshift and distance parameters will not be used in this function.')
    
    def SNR_error(redshift):
        redshift = redshift[0]
        if redshift < MIN_REDSHIFT:
            redshift = MIN_REDSHIFT
        distance = cosmology_model.luminosity_distance(redshift).value
        mod_params = params | {'redshift': redshift, 'luminosity_distance': distance}
        return np.log(compute_SNR(mod_params, detector)/target_SNR)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'The iteration is not making good progress')
        redshift, _, ier, _ = fsolve(
            func=SNR_error, 
            x0=0.01,
            full_output=True,
            maxfev=10000,
            )

    redshift = redshift[0]
    distance = cosmology_model.luminosity_distance(redshift).value
    
    if ier != 1:
        raise ValueError('Horizon computation did not converge!')

    return distance, redshift

def randomized_orientation_params(rng = DEFAULT_RNG):
    
    return {
        'theta_jn': np.arccos(rng.uniform(-1., 1.)),
        'dec': np.arccos(rng.uniform(-1., 1.)) - np.pi / 2.,
        'ra': rng.uniform(0, 2. * np.pi),
        'psi': rng.uniform(0, 2. * np.pi),
        'phase': rng.uniform(0, 2. * np.pi),
        'geocent_time': rng.uniform(1735257618, 1766793618) # full year 2035
    }

def horizon_varying_orientation(base_params: dict, samples: int, detector: Detector, **kwargs):
    
    distances = np.zeros(samples)
    redshifts = np.zeros(samples)
    
    for i in tqdm(range(samples)):
        params = base_params | randomized_orientation_params()
        distances[i], redshifts[i] = horizon(params, detector, **kwargs)
        
    return distances, redshifts
