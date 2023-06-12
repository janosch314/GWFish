import numpy as np
from GWFish.modules.detection import get_moon_coordinates, INTERP_MOON_POSIION, INTERP_MOON_RANGE

def test_coordinates():
    x, y, z = get_moon_coordinates(np.linspace(1e9, 1e9+3e7))
    
    assert INTERP_MOON_POSIION is not None
    assert INTERP_MOON_RANGE[0] < 1e9
    assert INTERP_MOON_RANGE[1] > 1e9 + 1e7