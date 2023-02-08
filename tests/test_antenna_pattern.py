import pytest
import numpy as np
from GWFish.modules.detection import Detector

@pytest.fixture
def detector_component():
    detector = Detector('ET', parameters=[None], fisher_parameters=[None])
    return detector.components[0]
    

@pytest.mark.xfail
def test_antenna_pattern_between_zero_and_one(detector_component):
    F_plus, F_cross = detector_component.antenna_pattern(ra=0., dec=0., gps_time=1359826914.)
    assert 0. <= F_plus <= 1.
    assert 0. <= F_cross <= 1.