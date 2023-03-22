import pytest
import numpy as np
from GWFish.modules.horizon import horizon, Detector, compute_SNR, MIN_REDSHIFT
from hypothesis import strategies as st
from hypothesis import given, settings, example
from datetime import timedelta

@st.composite
def extrinsic(draw):
    right_ascension = draw(
        st.floats(min_value=0, max_value=2 * np.pi),
    )
    declination = draw(
        st.floats(min_value=0, max_value=np.pi),
    )
    polarization = draw(
        st.floats(min_value=0, max_value=2 * np.pi),
    )
    gps_time = draw(
        st.floats(min_value=1.0, max_value=3786480018.0),  # 1980 to 2100
    )
    theta_jn = draw(
        st.floats(min_value=0., max_value=np.pi)
    )
    phase = draw(
        st.floats(min_value=0., max_value=2*np.pi)
    )
    return right_ascension, declination, polarization, gps_time, theta_jn, phase

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
    
    assert isinstance(distance, float)
    assert isinstance(redshift, float)
    
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

@pytest.mark.parametrize('detector_name', ['LGWA_Soundcheck', 'LGWA', 'LISA'])
@pytest.mark.parametrize('mass', [.6, 1e3, 1e7])
@given(extrinsic())
@settings(max_examples=4, deadline=timedelta(milliseconds=1000))
@example((
    2.94417698, 
    0.35331536, 
    5.85076693, 
    4.97215904, 
    2.43065638, 
    1.76231585e+09
))
def test_difficult_convergence_of_horizon_calculation(mass, detector_name, extrinsic):
    """A few examples of parameters for which there have 
    been problems in the past.
    """
    right_ascension, declination, polarization, gps_time, theta_jn, phase = extrinsic
    
    params = {
            'mass_1': mass,
            'mass_2': mass,
            'theta_jn': theta_jn, 
            'dec': declination, 
            'ra': right_ascension, 
            'psi': polarization, 
            'phase': phase, 
            'geocent_time': gps_time,
        }
    detector = Detector(detector_name, parameters= [None], fisher_parameters= [None])
    
    distance, redshift = horizon(params, detector)
    assert np.isclose(
        compute_SNR(
            params | {'redshift': redshift, 'luminosity_distance': distance}, 
            detector), 
        9, rtol=1e-3)

@pytest.mark.parametrize('detector_name', ['LGWA'])
@pytest.mark.parametrize('mass,equals_zero', [
    (1e7, False),
    (3e7, False),
    (1e9, True),
    (1e10, True)
])
@given(extrinsic())
@settings(max_examples=4, deadline=timedelta(milliseconds=500))
def test_horizon_for_very_large_masses(mass, equals_zero, detector_name, extrinsic):
    """Test horizon computation for large masses.
    
    For masses 5e6 <~ m <~ 3e7, the computed SNR is zero in some cases when the source is at high redshift,
    but at low redshift it is in band.
    
    For even larger masses, the computed SNR is zero even if the source is nearby. So,
    the horizon function should just return zero, and a warning.
    """
    right_ascension, declination, polarization, gps_time, theta_jn, phase = extrinsic
    
    params = {
            'mass_1': mass,
            'mass_2': mass,
            'theta_jn': theta_jn, 
            'dec': declination, 
            'ra': right_ascension, 
            'psi': polarization, 
            'phase': phase, 
            'geocent_time': gps_time,
        }
    detector = Detector(detector_name, parameters= [None], fisher_parameters= [None])
    
    if equals_zero:
        with pytest.warns():
            distance, redshift = horizon(params, detector)
            assert distance == 0.
    else:
        distance, redshift = horizon(params, detector)
        assert distance > 0.
        assert np.isclose(
            compute_SNR(
                params | {'redshift': redshift, 'luminosity_distance': distance}, 
                detector), 
            9, rtol=1e-3)
    
