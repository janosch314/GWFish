from datetime import timedelta
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from pycbc.detector import Detector as DetectorPycbc

from GWFish.modules.detection import Detector, Network
from GWFish.modules.horizon import (MIN_REDSHIFT, compute_SNR,
                                    compute_SNR_network, find_optimal_location,
                                    horizon, horizon_varying_orientation)

# TODO: change this according to https://docs.pytest.org/en/latest/example/parametrize.html#apply-indirect-on-particular-arguments

@pytest.fixture
def network():
    # return Network(detector_ids=['ET'])
    return Network(detector_ids=['ET', 'CE20'])

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
        st.floats(min_value=4e8, max_value=3786480018.0),  # 1992 to 2100
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
        'mass_1_source': 1.4,
        'mass_2_source': 1.4,
        'theta_jn': 5/6 * np.pi,
        'ra': 3.45,
        'dec': -0.41,
        'psi': 1.6,
        'phase': 0,
        'geocent_time': 1187008882, 
    }
    
    detector = Detector('LGWA')
    
    distance, redshift = horizon(params, detector)
    
    assert isinstance(distance, float)
    assert isinstance(redshift, float)
    
    params2 = params | {
        'mass_1_source': 2.8, 
        'mass_2_source': 2.8, 
    }
    
    distance2, redshift2 = horizon(params2, detector)
    
    assert np.isclose(distance2, 2*distance, rtol=2e-1)
    
def test_horizon_warns_when_given_redshift():
    params = {
        'redshift': 0.4,
        'mass_1_source': 1.4, 
        'mass_2_source': 1.4, 
        'theta_jn': 5/6 * np.pi,
        'ra': 3.45,
        'dec': -0.41,
        'psi': 1.6,
        'phase': 0,
        'geocent_time': 1187008882, 
    }
    
    detector = Detector('LGWA')

    with pytest.warns():
        distance, redshift = horizon(params, detector)

@pytest.mark.parametrize('detector_name', ['LGWA', 'LISA'])
@pytest.mark.parametrize('mass', [.6, 1e3, 1e7])
@given(extrinsic())
@settings(max_examples=2, deadline=timedelta(milliseconds=10000))
@example((
    2.94417698, 
    0.35331536, 
    5.85076693, 
    1.76231585e+09,
    2.43065638, 
    4.97215904, 
))
def test_difficult_convergence_of_horizon_calculation(mass, detector_name, extrinsic):
    """A few examples of parameters for which there have 
    been problems in the past.
    """
    right_ascension, declination, polarization, gps_time, theta_jn, phase = extrinsic
    
    params = {
            'mass_1_source': mass,
            'mass_2_source': mass,
            'theta_jn': theta_jn, 
            'dec': declination, 
            'ra': right_ascension, 
            'psi': polarization, 
            'phase': phase, 
            'geocent_time': gps_time,
        }
    detector = Detector(detector_name)
    
    distance, redshift = horizon(params, detector)
    assert distance > 0.
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
@settings(max_examples=3, deadline=timedelta(seconds=2))
def test_horizon_for_very_large_masses(mass, equals_zero, detector_name, extrinsic):
    """Test horizon computation for large masses.
    
    For masses 5e6 <~ m <~ 3e7, the computed SNR is zero in some cases when the source is at high redshift,
    but at low redshift it is in band.
    
    For even larger masses, the computed SNR is zero even if the source is nearby. So,
    the horizon function should just return zero, and a warning.
    """
    right_ascension, declination, polarization, gps_time, theta_jn, phase = extrinsic
    
    params = {
            'mass_1_source': mass,
            'mass_2_source': mass,
            'theta_jn': theta_jn, 
            'dec': declination, 
            'ra': right_ascension, 
            'psi': polarization, 
            'phase': phase, 
            'geocent_time': gps_time,
        }
    detector = Detector(detector_name)
    
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

@pytest.mark.parametrize('mass', [30.,])
@given(extrinsic())
@settings(
    max_examples=10, 
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
    deadline=timedelta(seconds=2)
)
def test_horizon_computation_with_network(mass, network, extrinsic):
    right_ascension, declination, polarization, gps_time, theta_jn, phase = extrinsic
    
    params = {
        'mass_1_source': mass,
        'mass_2_source': mass,
        'theta_jn': theta_jn, 
        'dec': declination, 
        'ra': right_ascension, 
        'psi': polarization, 
        'phase': phase, 
        'geocent_time': gps_time,
    }
    
    distance, redshift = horizon(params, network)
    assert distance > 0.
    assert np.isclose(
        compute_SNR_network(
            params | {'redshift': redshift, 'luminosity_distance': distance}, 
            network), 
        9, rtol=1e-3)
    
@pytest.mark.parametrize('mass', [30.,])
def test_optimal_parameter_finding(mass, network):
    base_params = {
        'mass_1_source': mass,
        'mass_2_source': mass,
        'theta_jn': 0., 
        'psi': 0., 
        'phase': 0., 
        'geocent_time': 1e6,
    }

    best_params = find_optimal_location(base_params, network)
    
    distance, redshift = horizon(best_params, network)
    
    distances, redshifts, parameters = horizon_varying_orientation(base_params, 5, network, return_parameters=True)
    
    assert np.all(distances < distance)
    assert np.all(redshifts < redshift)

@pytest.mark.skip('Old test, not really useful since there may be several optimums')
@pytest.mark.parametrize(
    ['detector_pycbc', 'detector_gwfish'],
    [
        ('V1', 'VIR')
    ])
@given(gps_time=st.floats(1e6, 1e10))
def test_optimal_parameter_finding_against_pycbc(detector_pycbc, detector_gwfish, gps_time):
    
    base_params = {
        'mass_1_source': 1.4,
        'mass_2_source': 1.4,
        'theta_jn': 0., 
        'psi': 0., 
        'phase': 0., 
        'geocent_time': gps_time,
    }

    detector = Detector(detector_gwfish)

    best_params = find_optimal_location(base_params, detector)
    best_params.pop('luminosity_distance')
    best_params.pop('redshift')

    ra_gwfish = best_params['ra']
    dec_gwfish = best_params['dec']
    
    detector_2 = DetectorPycbc(detector_pycbc)
    
    ra_pycbc, dec_pycbc = detector_2.optimal_orientation(gps_time)
    
    assert np.isclose(ra_gwfish, ra_pycbc)
    assert np.isclose(dec_gwfish, dec_pycbc)

@pytest.mark.skip('Feature not fully implemented yet')
@pytest.mark.parametrize(
    ['detector_name', 'bns_range'],
    [
        ('VIR_O2', 30),
        ('VIR', 260),
        ('LHO', 330),
        ('LLO', 330),
        ('KAG', 128),
    ]
)
def test_against_lrr_paper(detector_name, bns_range):
    base_params = {
        'mass_1_source': 1.4,
        'mass_2_source': 1.4,
        'theta_jn': 0., 
        'psi': 0., 
        'phase': 0., 
        'geocent_time': 0.,
    }

    detector = Detector(detector_name)

    best_params = find_optimal_location(base_params, detector)
    best_params.pop('luminosity_distance')
    best_params.pop('redshift')
    
    distance, redshift = horizon(best_params, detector, target_SNR=8)
    
    # this is the approximate peanut factor, but it can be calculated
    # in a better way: https://github.com/hsinyuc/distancetool/blob/08faf7c8e6ce86c44d33c498d3082f2b8b3b7d13/codes/find_horizon_range_de.py#L175
    peanut_factor = 2.264
    estimated_bns_range = distance / peanut_factor
    
    assert np.isclose(estimated_bns_range, bns_range, rtol=0.1)


@pytest.mark.parametrize('mass', [2000.,])
def test_horizon_with_network_against_single_detector(mass):
    params = {
        'mass_1_source': mass,
        'mass_2_source': mass,
        'theta_jn': 0., 
        'psi': 0., 
        'phase': 0., 
        'geocent_time': 1e9,
        'ra': 1.,
        'dec': 1.5
    }
    
    et_lgwa_network = Network(['ET', 'LGWA'])
    et_network = Network(['ET'])
    et_detector = Detector('ET')
    
    d1, z1 = horizon(params, et_detector)
    d2, z2 = horizon(params, et_network)
    d3, z3 = horizon(params, et_lgwa_network)
    
    assert np.isclose(d1, d2)
    assert np.isclose(z1, z2)
    
    # ET+LGWA should have a higher horizon than ET alone
    assert d1*1.1 < d3
    assert z1*1.1 < z3


@pytest.mark.skip('Very slow test')

@pytest.mark.parametrize('detector_name', ['ET', 'LGWA', 'VIR'])
def test_best_position_makes_sense(detector_name, plot):
    base_params = {
        'mass_1_source': 1.4,
        'mass_2_source': 1.4,
        'theta_jn': 0., 
        'psi': 0., 
        'phase': 0., 
        'geocent_time': 1_800_000_000,
    }

    detector = Detector(detector_name)

    best_params = find_optimal_location(base_params, detector)
    best_params['redshift'] = 0.
    best_params['luminosity_distance'] = 1.

    ra = best_params['ra']
    dec = best_params['dec']
    
    ra_grid = np.linspace(0, 2 * np.pi, num=40)
    dec_grid = np.linspace(-np.pi, np.pi, num=40)
    
    RA, DEC = np.meshgrid(ra_grid, dec_grid)
    
    snrs = np.zeros_like(RA)
    
    for (i, j), _ in np.ndenumerate(RA):
        
        snrs[i, j] = compute_SNR(best_params | {'ra': RA[i, j], 'dec': DEC[i, j]}, detector)
    
    snr_at_computed_max = compute_SNR(best_params | {'ra': ra, 'dec': dec}, detector)
    
    # tiny tolerance to account for numerical errors
    assert snr_at_computed_max * (1+1e-6) >= np.max(snrs) 
    
    if plot:
    
        plt.contourf(RA, DEC, snrs, levels=20)
        plt.xlabel('Right ascension')
        plt.ylabel('Declination')
        plt.scatter(ra, dec, c='black', s=100)
        plt.title(detector_name)
        plt.show()


@pytest.mark.skip('Very slow test')
@pytest.mark.parametrize('network_name', ['LGWA_ET'])
def test_best_position_makes_sense_network(network_name, plot):

    # around these masses the SNR for ET and LGWA are similar;
    
    base_params = {
        'mass_1_source': 4000.,
        'mass_2_source': 4000.,
        'theta_jn': 0., 
        'psi': 0., 
        'phase': 0., 
        'geocent_time': 1800000000,
    }

    network = Network(network_name.split('_'))

    best_params = find_optimal_location(base_params, network)
    best_params['redshift'] = 0.
    best_params['luminosity_distance'] = 1.

    ra = best_params['ra']
    dec = best_params['dec']
    
    ra_grid = np.linspace(0, 2 * np.pi, num=40)
    dec_grid = np.linspace(-np.pi, np.pi, num=40)
    
    RA, DEC = np.meshgrid(ra_grid, dec_grid)
    
    snrs = np.zeros_like(RA)
    
    for (i, j), _ in np.ndenumerate(RA):
        
        snrs[i, j] = compute_SNR_network(best_params | {'ra': RA[i, j], 'dec': DEC[i, j]}, network)
    
    snr_at_computed_max = compute_SNR_network(best_params | {'ra': ra, 'dec': dec}, network)
    
    # tiny tolerance to account for numerical errors
    # assert snr_at_computed_max >= np.max(snrs) 
    
    if plot:
    
        plt.contourf(RA, DEC, snrs, levels=20)
        plt.xlabel('Right ascension')
        plt.ylabel('Declination')
        plt.scatter(ra, dec, c='black', s=100)
        plt.title(network_name)
        plt.show()
