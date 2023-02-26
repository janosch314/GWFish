import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


def spherical_to_cartesian(radius, lat, lon):
    rcos_theta = radius * np.cos(lat)
    x = rcos_theta * np.cos(lon)
    y = rcos_theta * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    radius = np.sqrt(x**2 + y**2 + z**2)
    lon = np.arctan2(y, x)
    lat = np.arcsin(z / radius)

    return radius, lat, lon


def plot_antenna_pattern(function):
    proj = ccrs.Mollweide()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    # ax = plt.axes(projection=ccrs.Mollweide())

    nlon = 240
    nlat = 120
    lon_degrees = np.linspace(0, 360.0, nlon)
    lat_degrees = np.linspace(-89.0, 89.0, nlat)

    lon = np.deg2rad(lon_degrees)
    lat = np.deg2rad(lat_degrees)

    lon2d, lat2d = np.meshgrid(lon, lat)

    # F_plus for a detector at the North Pole :D
    data = function(lat2d, lon2d)

    mappable = ax.contourf(
        np.rad2deg(lon2d),
        np.rad2deg(lat2d),
        data,
        cmap=plt.get_cmap("seismic"),
        levels=np.linspace(-1, 1, num=50),
        transform=ccrs.PlateCarree(),  # not super clear from the docs,
        # but I think this just means "do not transform", since
        # the Plate-Carree is the equirectangular projection.
        # anyhow, do not remove it :D
        # transform_first=True,
    )
    ax.coastlines()
    ax.set_global()
    plt.colorbar(mappable, ax=ax)
    plt.savefig("antenna_pattern.png")
    plt.show()
