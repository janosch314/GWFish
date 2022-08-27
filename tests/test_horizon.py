import pytest
import numpy as np
from GWFish.modules.horizon import horizon, Detector, compute_SNR

def test_horizon_computation_result_170817_scaling():
    """
    Kind of a silly check: in the low-frequency regime, the 
    end of the signal is irrelevant, and we have a h ~ M scaling, 
    as well as h ~ 1 / d.

    Therefore, we'd expect that doubling the mass should roughly 
    double the horizon for that specific event.
    
    This also doubles as a smoke test. 
    """
    
    params = {
        'mass_1': 1.4,
        'mass_2': 1.4,
        'theta_jn': 5/6 * np.pi,
        'ra': 3.45,
        'dec': -0.41,
        'psi': 1.6,
        'phase': 0,
        'geocent_time': 1187008882, 
    }
    
    detector = Detector('LGWA', parameters= [None], fisher_parameters= [None])
    
    distance, redshift = horizon(params, detector)
    
    params2 = params | {
        'mass_1': 2.8, 
        'mass_2': 2.8, 
    }
    
    distance2, redshift2 = horizon(params2, detector)
    
    assert np.isclose(distance2, 2*distance, rtol=2e-1)
    
def test_horizon_warns_when_given_redshift():
    params = {
        'redshift': 0.4,
        'mass_1': 1.4, 
        'mass_2': 1.4, 
        'theta_jn': 5/6 * np.pi,
        'ra': 3.45,
        'dec': -0.41,
        'psi': 1.6,
        'phase': 0,
        'geocent_time': 1187008882, 
    }
    
    detector = Detector('LGWA', parameters= [None], fisher_parameters= [None])

    with pytest.warns():
        distance, redshift = horizon(params, detector)

@pytest.mark.xfail
@pytest.mark.parametrize('mass', [1e3, 1e4, 1e5, 1e6])
def test_difficult_convergence_of_horizon_calculation(mass):
    """A few examples of parameters for which there have 
    been problems in the past.
    """
    
    params = {
            'mass_1': mass,
            'mass_2': mass,
            'theta_jn': 2.94417698, 
            'dec': 0.35331536, 
            'ra': 5.85076693, 
            'psi': 4.97215904, 
            'phase': 2.43065638, 
            'geocent_time': 1.76231585e+09,
        }
    detector = Detector('LGWA', parameters= [None], fisher_parameters= [None])
    
    distance, redshift = horizon(params, detector)
    assert np.isclose(
        compute_SNR(
            params | {'redshift': redshift, 'luminosity_distance': distance}, 
            detector), 
        9, rtol=1e-3)

def test_randomized_horizon_computation():
    pass