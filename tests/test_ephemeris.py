from GWFish.modules import ephemeris
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

def time_execution(func, *args, **kwargs):
    t1 = perf_counter()
    func(*args, **kwargs)
    t2 = perf_counter()
    return t2 - t1

def test_ephemeris_caching():
    moon = ephemeris.MoonEphemeris()
    
    # about 3 years
    times = np.linspace(0, 1e8, num=10_000)
    
    first_time = time_execution(moon.get_coordinates, times)
    second_time = time_execution(moon.get_coordinates, times)
    
    # the first time should take longer, more than a tenth of a second
    # while for the second time we should already have cached the ephemeris
    # so it should take less than a millisecond (much less, even, but
    # this test can be a bit loose)
    assert first_time > 1e-1
    assert second_time < 1e-3
    

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
        
        plt.show()