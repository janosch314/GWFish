"""
This module is aimed at computing detection horizons.

The thing we want to compute is the luminosity distance at which a given
signal will be detected (with a given SNR).
"""

from typing import Union
import warnings
import numpy as np
from tqdm import tqdm

from astropy.cosmology import Planck18
import astropy.cosmology as cosmology
import astropy.units as u

from scipy.optimize import brentq, minimize

from .detection import SNR, Detector, projection, Network
from .waveforms import LALFD_Waveform

DEFAULT_RNG = np.random.default_rng(seed=1)

WAVEFORM_MODEL = 'IMRPhenomD'
MIN_REDSHIFT = 1e-20
MAX_REDSHIFT = 500

def compute_SNR(params: dict, detector: Detector, waveform_model: str = WAVEFORM_MODEL):

    data_params = {
        'frequencyvector': detector.frequencyvector,
        'f_ref': 50.
    }
    waveform_obj = LALFD_Waveform(waveform_model, params, data_params)
    polarizations = waveform_obj()
    timevector = waveform_obj.t_of_f
    
    signal = projection(
        params,
        detector,
        polarizations,
        timevector
    )
    
    component_SNRs = SNR(detector, signal)
    return np.sqrt(np.sum(component_SNRs**2))

def compute_SNR_network(params: dict, network: Network, waveform_model: str = WAVEFORM_MODEL):
    
    square_snrs = [
        compute_SNR(params, detector, waveform_model)**2
        for detector in network.detectors
    ]

    return np.sqrt(np.sum(square_snrs))

def horizon(
    params: dict,
    detector: Union[Detector, Network],
    target_SNR: int = 9, 
    waveform_model: str = WAVEFORM_MODEL,
    cosmology_model: cosmology.Cosmology = Planck18,
    source_frame_masses: bool = True,
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
    
    if isinstance(detector, Detector):
        snr_computer = compute_SNR
    elif isinstance(detector, Network):
        snr_computer = compute_SNR_network
    
    def SNR_error(redshift):
        distance = cosmology_model.luminosity_distance(redshift).value
        mod_params = params | {'luminosity_distance': distance}
        if source_frame_masses:
            mod_params['redshift'] = redshift
        with np.errstate(divide='ignore'):
            return np.log(snr_computer(mod_params, detector, waveform_model)/target_SNR)
    
    if not SNR_error(MIN_REDSHIFT) > 0:
        warnings.warn('The source is completely out of band')
        return 0., 0.
    
    redshift, r = brentq(SNR_error, MIN_REDSHIFT, MAX_REDSHIFT, full_output=True)
    if not r.converged:
        raise ValueError('Horizon computation did not converge!')
        
    distance = cosmology_model.luminosity_distance(redshift).value
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

def horizon_varying_orientation(base_params: dict, samples: int, detector: Union[Detector, Network], progressbar = True, return_parameters=False, **kwargs):
    
    distances = np.zeros(samples)
    redshifts = np.zeros(samples)
    parameters = []
    
    iterator = range(samples)
    if progressbar:
        iterator = tqdm(iterator)
    
    for i in iterator:
        params = base_params | randomized_orientation_params()
        distances[i], redshifts[i] = horizon(params, detector, **kwargs)
        if return_parameters:
            parameters.append(params.copy())
    
    if return_parameters:
        return distances, redshifts, parameters
        
    return distances, redshifts

def find_optimal_location(
    base_params: dict, 
    detector: Union[Detector, Network], 
    waveform_model: str = WAVEFORM_MODEL,
    ):
    """Determine optimal source location for a given detector or 
    network by maximizing the SNR.
    """
    
    if isinstance(detector, Detector):
        snr_computer = compute_SNR
    elif isinstance(detector, Network):
        snr_computer = compute_SNR_network

    params = base_params.copy()
    params['redshift'] = MIN_REDSHIFT
    params['luminosity_distance'] = 1e-15
    
    def make_params(x):
        ra, dec = x
        params['ra'] = ra
        params['dec'] = dec
        return params

    def to_minimize(x):
        return - snr_computer(make_params(x), detector, waveform_model)
    
    x0 = [0., 1.]
    if 'ra' in params:
        x0[0] = params['ra']
    
    if 'dec' in params:
        x0[1] = params['dec']
    
    res = minimize(
        fun=to_minimize, 
        x0=x0,
        bounds=[
            (0, 2*np.pi), 
            (0, np.pi),
        ]
    )

    return make_params(res.x)