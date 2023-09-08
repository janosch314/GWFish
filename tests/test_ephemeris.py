from GWFish.modules import ephemeris
from time import perf_counter
import numpy as np

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
    

