from astropy.coordinates import get_body_barycentric, ICRS, GCRS, EarthLocation
from astropy.time import Time
from scipy.interpolate import interp1d
import numpy as np
from abc import ABC, abstractmethod
import logging
import GWFish.modules.constants as cst

class EphemerisInterpolate:
    """This class provides a way to efficiently compute the xyz coordinates 
    of a body in the Solar System, as a function of time, by caching the ephemeris.
    """

    earliest_possible_time = 0. # gps time for ~1980
    interp_kind = 'linear'
    
    def __init__(self):
        self.interp_gps_time_range = (0,0)
        self.interp_gps_position = None

    @abstractmethod
    def get_icrs_from_times(self, times):
        ...

    @property
    def time_step_seconds(self):
        # time step of the saved ephemeris
        return 3600*12.

    def compute_xyz_cordinates(self, times):
        moon = self.get_icrs_from_times(times)
        moon.representation_type = 'cartesian'
        moon_x = moon.x.si.value
        moon_y = moon.y.si.value
        moon_z = moon.z.si.value
        return moon_x, moon_y, moon_z

    def create_position_interp(self, times):
        
        x, y, z = self.compute_xyz_cordinates(times)
        
        return (
            interp1d(times, x, bounds_error=False, fill_value=np.nan, kind=self.interp_kind),
            interp1d(times, y, bounds_error=False, fill_value=np.nan, kind=self.interp_kind),
            interp1d(times, z, bounds_error=False, fill_value=np.nan, kind=self.interp_kind),
        )

    def interpolation_not_computed(self, times):
        """This function will return True if the interpolation has not been computed yet,
        or if the times are outside of the range of the interpolation.
        Otherwise, it will return False.
        """

        if self.interp_gps_position is None:
            return True

        if (
            times[0] < self.interp_gps_time_range[0]
        ) and (
            self.interp_gps_time_range[0] > self.earliest_possible_time
            ):
            return True
        if times[-1] > self.interp_gps_time_range[1]:
            return True
        
        return False

    def get_coordinates(self, times):
        
        if self.interpolation_not_computed(times):
            logging.info('Computing interpolating object')

            t0, t1 = max(times[0], self.earliest_possible_time), times[-1]
            time_interval = t1 - t0
            self.interp_gps_time_range = t0 - time_interval / 10, t1 + time_interval / 10
            
            # ensure at least two points
            n_points = int(np.ceil(time_interval / self.time_step_seconds)) + 1

            new_times = np.linspace(*self.interp_gps_time_range, num=n_points)
            self.interp_gps_position = self.create_position_interp(new_times)

            logging.info('Finished computing interpolating object')
        
        interp_x, interp_y, interp_z = self.interp_gps_position
        return (
            interp_x(times), 
            interp_y(times), 
            interp_z(times),
        )

    def phase_term(self, ra, dec, timevector, frequencyvector):
    
        theta = np.pi/2. - dec
        
        kx_icrs = -np.sin(theta) * np.cos(ra)
        ky_icrs = -np.sin(theta) * np.sin(ra)
        kz_icrs = -np.cos(theta)

        x, y, z = self.get_coordinates(timevector)

        phase_shift = (
            x * kx_icrs +
            y * ky_icrs +
            z * kz_icrs
        ) * 2 * np.pi / cst.c * frequencyvector

class MoonEphemeris(EphemerisInterpolate):
    
    def get_icrs_from_times(self, times):
        return get_body_barycentric(
            "moon", 
            Time(times, format='gps'), 
            ephemeris='jpl'
        )

class EarthEphemeris(EphemerisInterpolate):
    
    @property
    def time_step_seconds(self):
        return 3600

    def get_icrs_from_times(self, times):
        return get_body_barycentric(
            "earth", 
            Time(times, format='gps'), 
            ephemeris='jpl'
        )

class EarthLocationEphemeris(EphemerisInterpolate):
    
    def __init__(self, location: EarthLocation):
        super().__init__()
        
        self.location = location
    
    @property
    def time_step_seconds(self):
        return 1800.

    def get_icrs_from_times(self, times):
        
        time = Time(times, format='gps')
        
        obslocation = self.location.get_gcrs(time)
        # obslocation.representation_type = 'cartesian'
        
        earth = get_body_barycentric(
            "earth", 
            time, 
            ephemeris='jpl'
        )
    
        return earth + obslocation.data