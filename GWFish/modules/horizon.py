"""
This module is aimed at computing detection horizons.

The thing we want to compute is the luminosity distance at which a given
signal will be detected (with a given SNR).
"""


from typing import Union, TYPE_CHECKING
if TYPE_CHECKING:
    import pandas as pd
import warnings
import numpy as np
from tqdm import tqdm

from astropy.cosmology import Planck18
import astropy.cosmology as cosmology
import astropy.units as u

from scipy.optimize import brentq, minimize, dual_annealing

from .detection import SNR, Detector, projection, Network
from .waveforms import LALFD_Waveform, DEFAULT_WAVEFORM_MODEL, Waveform

DEFAULT_RNG = np.random.default_rng(seed=1)

MIN_REDSHIFT = 1e-20
MAX_REDSHIFT = 1e6

def compute_SNR(
    params: "Union[dict[str, float], pd.DataFrame]", 
    detector: Detector, 
    waveform_model: str = DEFAULT_WAVEFORM_MODEL,
    waveform_class: type(Waveform) = LALFD_Waveform,
    redefine_tf_vectors: bool = False) -> float:
    """Compute the SNR for a single signal, and a single detector.
    
    :param params: parameters for the signal
    :param detector: detector to use
    :param waveform_model: waveform model to use - refer to [choosing an approximant](../how-to/choosing_an_approximant.md)
    :param waveform_class: waveform class to use - refer to [choosing an approximant](../how-to/choosing_an_approximant.md)
    :param redefine_tf_vectors: whether to redefine the time and frequency vectors 
    
    :return: the SNR
    """

    data_params = {
        'frequencyvector': detector.frequencyvector,
        'f_ref': 50.
    }
    waveform_obj = waveform_class(waveform_model, params, data_params)
    polarizations = waveform_obj()
    timevector = waveform_obj.t_of_f
    
    args = (params, detector, polarizations, timevector)
    
    if redefine_tf_vectors:
        signal, timevector, frequencyvector = projection(*args, redefine_tf_vectors=True)
    else:
        signal = projection(*args)
        frequencyvector = detector.frequencyvector

    component_SNRs = SNR(detector, signal, frequencyvector=np.squeeze(frequencyvector))
    return np.sqrt(np.sum(component_SNRs**2))

def compute_SNR_network(
    params: "Union[dict[str, float], pd.DataFrame]", 
    network: Network, 
    waveform_model: str = DEFAULT_WAVEFORM_MODEL,
    waveform_class: type(Waveform) = LALFD_Waveform,
    redefine_tf_vectors: bool = False
    ) -> float:
    
    square_snrs = [
        compute_SNR(params, detector, waveform_model, waveform_class, redefine_tf_vectors)**2
        for detector in network.detectors
    ]

    return np.sqrt(np.sum(square_snrs))

def horizon(
    params: dict,
    detector: Union[Detector, Network],
    target_SNR: float = 9., 
    waveform_model: str = DEFAULT_WAVEFORM_MODEL,
    waveform_class: type(Waveform) = LALFD_Waveform,
    cosmology_model: cosmology.Cosmology = Planck18,
    source_frame_masses: bool = True,
    redefine_tf_vectors: bool = False,
    ):
    """
    Given the parameters for a GW signal and a detector, this function 
    computes the luminosity distance and corresponding redshift 
    (as connected by a given cosmology) which the signal would 
    need to be at in order to have a given SNR - by default, 9.
    
    :param params: fixed parameters for the signal (should not include redshift or luminosity distance)
    :param detector: `Detector` or `Network` object to compute the horizon for
    :param target_SNR: the function will compute the distance such the signal will have this SNR in the given detector
    :param waveform_model: waveform model to use - refer to [choosing an approximant](../how-to/choosing_an_approximant.md)
    :param waveform_model: waveform class to use - refer to [choosing an approximant](../how-to/choosing_an_approximant.md)
    :param cosmology_model: (astropy) cosmology model to use to relate the redshift to the luminosity distance
    :param source_frame_masses: whether to assume the given mass is in the source frame, therefore it needs to be redshifted in order to compute the detected one. Default is True. 
    :param redefine_tf_vectors: whether to redefine the time and frequency vectors (useful when computing horizons for sources with very small frequency variation). Default is False.
    
    :return:
    
    - luminosity_distance in Mpc, 
    - corresponding redshift
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
            return np.log(snr_computer(mod_params, detector, waveform_model, waveform_class, redefine_tf_vectors)/target_SNR)
    
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
    waveform_model: str = DEFAULT_WAVEFORM_MODEL,
    waveform_class: type(Waveform) = LALFD_Waveform,
    redefine_tf_vectors: bool = False,
    **minimizer_kwargs,
    ):
    """Determine optimal source location for a given detector or 
    network by maximizing the SNR.
    """
    
    if isinstance(detector, Detector):
        snr_computer = compute_SNR
    elif isinstance(detector, Network):
        snr_computer = compute_SNR_network

    params = base_params.copy()
    params['redshift'] = 0.
    # ensure luminosity distance is very small
    params['luminosity_distance'] = 1e-15
    
    def make_params(x):
        ra, dec = x
        params['ra'] = ra
        params['dec'] = dec
        return params

    def to_minimize(x):
        return - snr_computer(make_params(x), detector, waveform_model, waveform_class, redefine_tf_vectors)

    x0 = [0., 1.]
    if 'ra' in params:
        x0[0] = params['ra']
    
    if 'dec' in params:
        x0[1] = params['dec']
    
    if 'maxiter' not in minimizer_kwargs:
        minimizer_kwargs['maxiter'] = 100
    
    res = dual_annealing(
        func=to_minimize, 
        bounds=[
            (0, 2*np.pi), 
            (-np.pi, np.pi),
        ],
        x0=x0,
        **minimizer_kwargs
    )

    del params['redshift']
    del params['luminosity_distance']
    
    return make_params(res.x)