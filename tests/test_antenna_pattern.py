import pytest
import numpy as np
from GWFish.modules.detection import Detector, GreenwichMeanSiderealTime
from hypothesis import strategies as st
from hypothesis import given, HealthCheck, settings, assume
from scipy.spatial.transform import Rotation

from .plots_and_transformations import (
    spherical_to_cartesian,
    cartesian_to_spherical,
    plot_antenna_pattern,
    plot_antenna_pattern_equirectangular,
)


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


@given(
    x=st.floats(-10, 10),
    y=st.floats(-10, 10),
    z=st.floats(-10, 10),
)
def test_spherical_cartesian(x, y, z):
    r, lat, lon = cartesian_to_spherical(x, y, z)
    assume(r > 0)
    xnew, ynew, znew = spherical_to_cartesian(r, lat, lon)

    assert np.isclose(x, xnew)
    assert np.isclose(y, ynew)
    assert np.isclose(z, znew)


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


def theoretical_antenna_pattern_plus(theta, phi, psi):
    return 0.5 * (1 + np.cos(theta) ** 2) * np.cos(2 * phi) * np.cos(2 * psi) - np.cos(
        theta
    ) * np.sin(2 * phi) * np.sin(2 * psi)


def theoretical_antenna_pattern_cross(theta, phi, psi):
    return -0.5 * (1 + np.cos(theta) ** 2) * np.cos(2 * phi) * np.sin(2 * psi) - np.cos(
        theta
    ) * np.sin(2 * phi) * np.cos(2 * psi)


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

    det_position = detector_component.position
    det_lon = detector_component.lon
    det_lat = detector_component.lat
    det_opening_angle = detector_component.opening_angle
    det_azimuth = detector_component.arm_azimuth

    assert np.isclose(det_opening_angle, np.pi / 2)

    rotation = Rotation.from_euler("zyx", [det_lon, det_lat, det_azimuth])

    gmst = GreenwichMeanSiderealTime(gps_time)
    theta = np.pi / 2.0 - declination
    phi = right_ascension - gmst
    psi_detector = polarization

    sky_xyz = spherical_to_cartesian(1, lat=declination, lon=right_ascension - gmst)

    detector_xyz = rotation.apply(np.array(sky_xyz))

    _, lat_detector, lon_detector = cartesian_to_spherical(*detector_xyz)

    theta_detector = np.pi / 2 - lat_detector
    phi_detector = lon_detector

    # but this is a different coordinate system!
    # these formulas give the antenna pattern as a function of
    # the two angles theta, phi of the source in the frame of the detector
    # while here, they are in the frame of the Earth

    # so here there should be a rotation by detector_component.

    F_plus_theoretical = theoretical_antenna_pattern_plus(
        theta_detector, phi_detector, psi_detector
    )
    F_cross_theoretical = theoretical_antenna_pattern_cross(
        theta_detector, phi_detector, psi_detector
    )

    assert np.isclose(F_plus_theoretical, F_plus)
    assert np.isclose(F_cross_theoretical, F_cross)


if __name__ == "__main__":

    def fplus(lat, lon):
        return 1 / 2 * (1 + np.cos(lon) ** 2) * np.cos(2 * lat)

    def fplus(lat, lon):
        return theoretical_antenna_pattern_plus(theta=np.pi / 2.0 - lat, phi=lon, psi=0)

    plot_antenna_pattern_equirectangular(fplus)
    plot_antenna_pattern(fplus)
