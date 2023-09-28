from GWFish.modules import ephemeris
from GWFish.modules.waveforms import t_of_f_PN
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation
import pytest

def time_execution(func, *args, **kwargs):
    t1 = perf_counter()
    func(*args, **kwargs)
    t2 = perf_counter()
    return t2 - t1

def test_ephemeris_caching():
    moon = ephemeris.MoonEphemeris()
    
    # about 30 years
    times = np.linspace(0, 1e9, num=10_000)
    
    first_time = time_execution(moon.get_coordinates, times)
    second_time = time_execution(moon.get_coordinates, times)
    
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
    
    x_moon, y_moon, z_moon = moon.get_coordinates(times)
    x_earth, y_earth, z_earth = earth.get_coordinates(times)
    
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
    
    x_loc, y_loc, z_loc = loc.get_coordinates(times)
    x_earth, y_earth, z_earth = earth.get_coordinates(times)
    
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
    
    x, y, z = ephem.get_coordinates(times)
    
    r = np.sqrt(x**2 + y**2 + z**2)
    
    
    assert not np.isnan(r).any()
    
    # the radius should be roughly an astronomical unit
    assert np.allclose(r, 150e9, rtol=5e-2)
    

@pytest.mark.parametrize('ephem', [
    ephemeris.EarthEphemeris(),
    ephemeris.EarthLocationEphemeris(EarthLocation.of_site('V1')),
    ephemeris.MoonEphemeris(),
])
def test_phase_term_differentiated(ephem, plot):

    frequencies = np.geomspace(0.1, 100., num=1000)
    times = t_of_f_PN({
            'geocent_time': 1.8e9,
            'mass_1': 1.4,
            'mass_2': 1.4,
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