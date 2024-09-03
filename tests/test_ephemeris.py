from copy import deepcopy
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.coordinates import EarthLocation

from GWFish.modules import ephemeris
from GWFish.modules.detection import Detector
from GWFish.modules.fishermatrix import compute_detector_fisher
from GWFish.modules.waveforms import t_of_f_PN


def time_execution(func, *args, **kwargs):
    t1 = perf_counter()
    func(*args, **kwargs)
    t2 = perf_counter()
    return t2 - t1

def test_ephemeris_caching():
    moon = ephemeris.MoonEphemeris()
    
    # about 30 years
    times = np.linspace(0, 1e9, num=10_000)
    
    first_time = time_execution(moon.get_coordinates, times, (0, 0, 0))
    second_time = time_execution(moon.get_coordinates, times, (0, 0, 0))
    
    # the first time should take longer, more than a tenth of a second
    # while for the second time we should already have cached the ephemeris
    # so it should take less than a millisecond (much less, even, but
    # this test can be a bit loose)
    assert first_time > 1e-1
    assert second_time < 1e-2
    

def test_earth_vs_moon(plot):
    
    moon = ephemeris.MoonEphemeris()
    earth = ephemeris.EarthEphemeris()
    
    times = np.linspace(0, 3600*24*60, num=10_000)
    
    x_moon, y_moon, z_moon = moon.get_coordinates(times, (0, 0, 0))
    x_earth, y_earth, z_earth = earth.get_coordinates(times, (0, 0, 0))
    
    r_x, r_y, r_z = x_moon - x_earth, y_moon - y_earth, z_moon - z_earth
    
    r = np.sqrt(r_x**2 + r_y**2 + r_z**2)
    
    # first sanity check: Earth - Moon distance is reasonable 
    # within a large tolerance, since it varies from perigee to apogee
    assert np.allclose(r, 384e6, rtol=1e-1)
    
    if plot:
        plt.plot(times/(3600*24*30), r_x/384e6)
        plt.plot(times/(3600*24*30), r_y/384e6)
        plt.plot(times/(3600*24*30), r_z/384e6)
        
        plt.xlabel('Time [months]')
        plt.ylabel('Distance [normalized to avg Earth/Moon distance]')
        plt.show()
        
def test_earth_center_vs_location(plot):
    
    earth = ephemeris.EarthEphemeris()
    loc = ephemeris.EarthLocationEphemeris(EarthLocation.of_site('V1'))
    
    times = np.linspace(0, 3600*24*2, num=10_000)
    
    x_loc, y_loc, z_loc = loc.get_coordinates(times, (0, 0, 0))
    x_earth, y_earth, z_earth = earth.get_coordinates(times, (0, 0, 0))
    
    r_x, r_y, r_z = x_loc - x_earth, y_loc - y_earth, z_loc - z_earth
    
    r = np.sqrt(r_x**2 + r_y**2 + r_z**2)
    
    # first sanity check: Earth - surface distance is reasonable 
    # within a large tolerance, since it varies from perigee to apogee
    # breakpoint()
    assert np.allclose(r, 6378e3, rtol=1e-2)
    
    if plot:
        plt.plot(times/(3600*24), r_x/6378e3)
        plt.plot(times/(3600*24), r_y/6378e3)
        plt.plot(times/(3600*24), r_z/6378e3)
        plt.plot(times/(3600*24), r/6378e3, label='Center of the Earth to location')
        
        plt.xlabel('Time [days]')
        plt.ylabel('Distance [normalized to Earth radius]')
        plt.show()
        

@pytest.mark.parametrize('ephem', [
    ephemeris.EarthEphemeris(),
    ephemeris.EarthLocationEphemeris(EarthLocation.of_site('V1')),
    ephemeris.MoonEphemeris(),
])
def test_all_coordinates_are_solar_centered(ephem):
    
    # a day
    times = np.linspace(0, 3600*24, num=10_000)
    
    x, y, z = ephem.get_coordinates(times, (0, 0, 0))
    
    r = np.sqrt(x**2 + y**2 + z**2)
    
    
    assert not np.isnan(r).any()
    
    # the radius should be roughly an astronomical unit
    assert np.allclose(r, 150e9, rtol=5e-2)


def test_geocentered_coordinates_are():
    
    ephem = ephemeris.EarthLocationGCRSEphemeris(EarthLocation.of_site('V1'))
    
    # a day
    times = np.linspace(0, 3600*24, num=10_000)
    
    x, y, z = ephem.get_coordinates(times, (0, 0, 0))
    
    r = np.sqrt(x**2 + y**2 + z**2)
    
    
    assert not np.isnan(r).any()
    
    # the radius should be roughly the one of the Earth
    assert np.allclose(r, 6.378e6, rtol=1e-2)


@pytest.mark.parametrize('ephem', [
    ephemeris.EarthEphemeris(),
    ephemeris.EarthLocationEphemeris(EarthLocation.of_site('V1')),
    ephemeris.MoonEphemeris(),
])
def test_phase_term_differentiated(ephem, plot):

    frequencies = np.geomspace(0.1, 2000., num=1000)
    times = t_of_f_PN({
            'geocent_time': 1.8e9,
            'mass_1_source': 1.4,
            'mass_2_source': 1.4,
            'redshift': 0.}, 
        frequencies)
    
    
    if plot:
        
        for interp_kind in [1, 3]:
            ephem.interp_kind = interp_kind
            ephem.__init__()
            baseline = ephem.phase_term(1., 1., times, frequencies)
            delta_ra = 1e-5
            phase_derivative_approx = (
            ephem.phase_term(1.+delta_ra, 1., times, frequencies) - 
            baseline) / delta_ra
            
            plt.semilogx(frequencies, phase_derivative_approx, label=interp_kind)
        plt.legend()
        plt.title(ephem.__class__.__name__)
        plt.show()
    

@pytest.mark.xfail
def test_einstein_telescope_localization(plot, gw170817_params):
    et = Detector('ET')
    
    location = et.components[0].ephem.location
    
    et_gcrs = Detector('ET')
    for component in et_gcrs.components:
        component.ephem = ephemeris.EarthLocationGCRSEphemeris(location)
    
    fisher_params = gw170817_params.columns.tolist()
    
    fisher_et, snr_et = compute_detector_fisher(
        et, 
        gw170817_params.iloc[0], 
        fisher_parameters=fisher_params,
        waveform_model='IMRPHenomD_NRTidalv2'
    )
    fisher_gcrs, snr_gcrs = compute_detector_fisher(
        et_gcrs, 
        gw170817_params.iloc[0], 
        fisher_parameters=fisher_params,
        waveform_model='IMRPHenomD_NRTidalv2'
    )
    
    assert np.isclose(snr_et, snr_gcrs)
    
    inverse_et = np.linalg.inv(fisher_et)
    inverse_gcrs = np.linalg.inv(fisher_gcrs)


@pytest.mark.xfail
@pytest.mark.parametrize('ephem', [
    ephemeris.EarthEphemeris(),
    ephemeris.EarthLocationEphemeris(EarthLocation.of_site('V1')),
    ephemeris.MoonEphemeris(),
])
def test_signal_ends_with_stationary_phase(ephem, plot):

    frequencies = np.geomspace(0.1, 2000., num=1000)
    times = t_of_f_PN({
            'geocent_time': 1.8e9,
            'mass_1_source': 1.4,
            'mass_2_source': 1.4,
            'redshift': 0.}, 
        frequencies)

    phase = ephem.phase_term(1., 1., times, frequencies)
    
    assert np.isclose(phase[-1], 0.)
    assert np.isclose(np.gradient(phase)[-1], 0.)