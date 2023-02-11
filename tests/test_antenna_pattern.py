import pytest
import numpy as np
from GWFish.modules.detection import Detector, GreenwichMeanSiderealTime
from hypothesis import strategies as st
from hypothesis import given, HealthCheck, settings


@pytest.fixture
def detector_component():
    detector = Detector("VIR", parameters=[None], fisher_parameters=[None])
    return detector.components[0]


@st.composite
def coordinates(draw):

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
    return right_ascension, declination, polarization, gps_time



@given(coordinates=coordinates())
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
def test_antenna_pattern_between_zero_and_one(detector_component, coordinates):

    right_ascension, declination, polarization, gps_time = coordinates

    F_plus, F_cross = detector_component.antenna_pattern(
        ra=right_ascension,
        dec=declination,
        psi=polarization,
        gps_time=gps_time,
    )
    assert -1.0 <= F_plus <= 1.0
    assert -1.0 <= F_cross <= 1.0


@pytest.mark.xfail
@settings(suppress_health_check=(HealthCheck.function_scoped_fixture,))
@given(coordinates=coordinates())
def test_antenna_pattern_against_analytical_formula(detector_component, coordinates):
    right_ascension, declination, polarization, gps_time = coordinates

    F_plus, F_cross = detector_component.antenna_pattern(
        ra=right_ascension,
        dec=declination,
        psi=polarization,
        gps_time=gps_time,
    )

    gmst = GreenwichMeanSiderealTime(gps_time)
    theta = np.pi / 2.0 - declination
    phi = right_ascension - gmst
    psi = polarization
    
    # but this is a different coordinate system!
    # these formulas give the antenna pattern as a function of 
    # the two angles theta, phi of the source in the frame of the detector
    # while here, they are in the frame of the Earth
    
    # so here there should be a rotation by detector_component.

    F_plus_theoretical = .5 * (1 + np.cos(theta) ** 2) * np.cos(2 * phi) * np.cos(
        2 * psi
    ) - np.cos(theta) * np.sin(2 * phi) * np.sin(2 * psi)
    F_cross_theoretical = - .5 * (1 + np.cos(theta) ** 2) * np.cos(2 * phi) * np.sin(
        2 * psi
    ) - np.cos(theta) * np.sin(2 * phi) * np.cos(2 * psi)

    assert np.isclose(F_plus_theoretical, F_plus)
    assert np.isclose(F_cross_theoretical, F_cross)