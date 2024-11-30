from astropy.coordinates import get_body_barycentric, ICRS, GCRS, EarthLocation
from astropy.time import Time
from scipy.interpolate import interp1d
import numpy as np
from abc import ABC, abstractmethod
import logging
import GWFish.modules.constants as cst
import warnings

class EphemerisInterpolate:
    """This class provides a way to efficiently compute the xyz coordinates 
    of a body in the Solar System, as a function of time, by caching the ephemeris.
    """

    earliest_possible_time = -1000_000_000. # gps time for ~1980
    interp_kind = 1
    
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*dubious year \(Note \d\)")
            body = self.get_icrs_from_times(times)
            
        body.representation_type = 'cartesian'
        body_x = body.x.si.value
        body_y = body.y.si.value
        body_z = body.z.si.value
        return body_x, body_y, body_z
    
    def create_position_interp(self, times):
        
        x, y, z = self.compute_xyz_cordinates(times)
        
        return (
            interp1d(times, x, bounds_error=False, fill_value=np.nan, kind=self.interp_kind),
            interp1d(times, y, bounds_error=False, fill_value=np.nan, kind=self.interp_kind),
            interp1d(times, z, bounds_error=False, fill_value=np.nan, kind=self.interp_kind),
        )

    def create_velocity_interp(self, times):

        x, y, z = self.compute_xyz_cordinates(times)
        
        v_x = np.gradient(x, times, edge_order=2)
        v_y = np.gradient(y, times, edge_order=2)
        v_z = np.gradient(z, times, edge_order=2)
        
        return (
            interp1d(times, v_x, bounds_error=False, fill_value=0, kind=self.interp_kind),
            interp1d(times, v_y, bounds_error=False, fill_value=0, kind=self.interp_kind),
            interp1d(times, v_z, bounds_error=False, fill_value=0, kind=self.interp_kind),
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

    def get_coordinates(self, times, center):
        
        if self.interpolation_not_computed(times):
            logging.info('Computing interpolating object')

            t0, t1 = max(times[0], self.earliest_possible_time), times[-1]
            if t1 <= self.earliest_possible_time:
                raise ValueError('Signal must end after 1980 (gps time=0)')
            time_interval = t1 - t0
            if time_interval < self.time_step_seconds:
                time_interval = self.time_step_seconds
            self.interp_gps_time_range = t0 - time_interval, t1 + time_interval
            
            # ensure at least three points for linear interpolation, five for cubic
            n_points = int(np.ceil(time_interval / self.time_step_seconds)) + self.interp_kind + 1
            # breakpoint()
            new_times = np.linspace(*self.interp_gps_time_range, num=n_points)

            self.interp_gps_position = self.create_position_interp(new_times)
            self.interp_gps_velocity = self.create_velocity_interp(new_times)

            logging.info('Finished computing interpolating object')
        
        interp_x, interp_y, interp_z = self.interp_gps_position
        return (
            interp_x(times) - center[0], 
            interp_y(times) - center[1], 
            interp_z(times) - center[2],
        )
        
    
    def get_velocity(self, times):
        
        self.get_coordinates([times[0], times[-1]], (0,0,0))
        
        interp_v_x, interp_v_y, interp_v_z = self.interp_gps_velocity
        return (
            interp_v_x(times),
            interp_v_y(times),
            interp_v_z(times),
        )
    
    def get_velocity_along_direction(self, times, direction):
        
        v_x, v_y, v_z = self.get_velocity(times)
        
        return (
            v_x * direction[0] / 299_792_458 +
            v_y * direction[1] / 299_792_458 +
            v_z * direction[2] / 299_792_458
        )
    
    def phase_term(self, ra, dec, timevector, frequencyvector, center):
    
        theta = np.pi/2. - dec
        
        kx_icrs = -np.sin(theta) * np.cos(ra)
        ky_icrs = -np.sin(theta) * np.sin(ra)
        kz_icrs = -np.cos(theta)

        x, y, z = self.get_coordinates(timevector, center)

        return (
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
        return 1800.

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
    
class EarthLocationGCRSEphemeris(EphemerisInterpolate):
    
    def __init__(self, location: EarthLocation):
        super().__init__()
        
        self.location = location
    
    @property
    def time_step_seconds(self):
        return 180.

    def get_icrs_from_times(self, times):
        
        time = Time(times, format='gps')
        
        obslocation = self.location.get_gcrs(time)
            
        return obslocation.data


class LISAEphemeris(EphemerisInterpolate):
    
    @property
    def time_step_seconds(self):
        return 1800.

    def get_icrs_from_times(self, times):
        
        time = Time(times, format='gps')
        
        obslocation = self.location.get_gcrs(time)
            
        return obslocation.data

    def compute_xyz_cordinates(self, times):
        return np.zeros_like(times), np.zeros_like(times), np.zeros_like(times)