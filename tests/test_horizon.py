import pytest
import numpy as np
from GWFish.modules.horizon import horizon, Detector

def test_horizon_computation_result_170817_scaling():
    """
    Kind of a silly check: in the low-frequency regime, the 
    end of the signal is irrelevant, and we have a h ~ M scaling, 
    as well as h ~ 1 / d.

    Therefore, we'd expect that doubling the mass should roughly 
    double the horizon for that specific event.
    
    This also doubles as a smoke test. 
    """
    
    one = np.ones((1,))
    params = {
        'mass_1': 1.4 * one,
        'mass_2': 1.4 * one,
        'theta_jn': 5/6 * np.pi * one,
        'ra': 3.45 * one,
        'dec': -0.41 * one,
        'psi': 1.6 * one,
        'phase': 0 * one,
        'geocent_time': 1187008882 * one, 
    }
    
    detector = Detector('LGWA', parameters= [None], fisher_parameters= [None])
    
    distance, redshift = horizon(params, detector)
    
    params2 = params | {
        'mass_1': 2.8 * one, 
        'mass_2': 2.8 * one, 
    }
    
    distance2, redshift2 = horizon(params2, detector)
    
    assert np.isclose(distance2, 2*distance, rtol=2e-1)
    
def test_horizon_warns_when_given_redshift():
    one = np.ones((1,))
    params = {
        'redshift': 0.4,
        'mass_1': 1.4 * one, 
        'mass_2': 1.4 * one, 
        'theta_jn': 5/6 * np.pi * one,
        'ra': 3.45 * one,
        'dec': -0.41 * one,
        'psi': 1.6 * one,
        'phase': 0 * one,
        'geocent_time': 1187008882 * one, 
    }
    
    detector = Detector('LGWA', parameters= [None], fisher_parameters= [None])

    with pytest.warns():
        distance, redshift = horizon(params, detector)
