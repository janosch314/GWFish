(parameter-definitions-units)=
# Parameter definitions and units

The parameters used to describe a CBC within `GWFish` are as follows:

- `'mass_1'`: the mass of the primary object, in solar masses $M_{\odot}$;
- `'mass_2'`: the mass of the secondary object, in solar masses $M_{\odot}$;
- `'redshift'`: the redshift of the merger;
- `'luminosity_distance'`: the luminosity distance of the merger, in Megaparsecs;
- `'theta_jn'`: the inclination angle (between the observation direction 
    and the angular momentum of the merger), in radians;
- `'psi'`: the polarization angle of the merger, in radians;
- `'phase'`: the initial phase of the waveform, in radians;
- `'ra'`: the right ascension of the merger, in radians;
- `'dec'`: the declination of the merger, in radians;
- `'geocent_time'`: the time of merger, expressed as [GPS time](https://www.andrews.edu/~tzs/timeconv/timeconvert.php?);
- `'max_frequency'`: a maximum frequency at which to cut off the waveform, in Hz;
- `'a_1'`: spin of the primary object, in units of the square of the mass 
    (often denoted as $\chi = J / M^2$ in $c=G=1$ natural units), 
    dimensionless (and $\in [-1, 1]$), 
    not available for all approximants;
- `'a_2'`: spin of the secondary object, not available for all approximants.
- `'lambda_1'`: tidal polarizability $\Lambda_1$ of the primary (compact) star; 
    for details on the definition see e.g.
    section III.D of [the GW170817 properties paper](https://arxiv.org/abs/1805.11579). This parameter is not available for all approximants;
- `'lambda_2'`: tidal polarizability $\Lambda_2$ of the secondary (compact) star,
    not available for all approximants.

```{warning}
Including parameters which do not apply for a specific approximants (e.g. giving `'a_1'` 
for `'gwfish_TaylorF2'`) will not currently raise an error, 
the parameters will be simply ignored.
```