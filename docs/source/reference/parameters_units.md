(parameter-definitions-units)=
# Parameter definitions and units

The parameters used to describe a {term}`CBC` within `GWFish` are as follows:

- `'chirp_mass'`: chirp mass of the binary in detector frame in  $M_{\odot}$;
- `'mass_ratio'`: ratio of the secondary mass over the primary mass, so that it ranges in $[0,1]$;
- `'luminosity_distance'`: the luminosity distance of the merger, in Mpc;
- `'theta_jn'`: the angle between the observation direction $\vec{N}$
    and the total angular momentum of the merger - the sum of orbital angular momentum and the compact objects' spins, $\vec{J} = \vec{L} + \vec{S}_1 +\vec{S}_2 + (\text{GR corrections})$, as opposed to $\iota$, which is the inclination angle between the observation direction and the orbital angular momentum $\vec{L}$ -, in radians;
- `'psi'`: the polarization angle of the merger, in radians;
- `'phase'`: the initial phase of the waveform, in radians;
- `'ra'`: the right ascension of the merger, in radians;
- `'dec'`: the declination of the merger, in radians;
- `'geocent_time'`: the time of merger, expressed as [GPS time](https://www.andrews.edu/~tzs/timeconv/timeconvert.php?);
- `'max_frequency_cutoff'`: a maximum frequency at which to cut off the waveform, in Hz;
- `'a_1'`: dimensionless spin parameter of primary component; it ranges in $[0, 1]$
- `'a_2'`: dimensionless spin parameter of secondary component; it ranges in $[0, 1]$
- `'tilt_1'`: zenith angle between the spin and orbital angular momenta for the primary component in [rad]; it ranges in $[0, \pi]$
- `'tilt_2'`: zenith angle between the spin and orbital angular momenta for the secondary component in [rad]; it ranges in $[0, \pi]$
- `'phi_12'`: difference between total and orbital angular momentum azimuthal angles in [rad]; it ranges in $[0, 2\pi]$
- `'phi_jl'`: difference between the azimuthal angles of the individual spin vector projections on to the orbital plane in [rad]; it ranges in $[0, 2\pi]$
- `'lambda_1'`: tidal polarizability $\Lambda_1$ of the primary (compact) star; 
    for details on the definition see e.g.
    section III.D of [the GW170817 properties paper](https://arxiv.org/abs/1805.11579). This parameter is not available for all approximants;
- `'lambda_2'`: tidal polarizability $\Lambda_2$ of the secondary (compact) star,
    not available for all approximants.

[!WARNING]
There are different combinations of masses that can be passed as input in `GWFish`:
- `chirp_mass`, `mass_ratio`: as described above
- `chirp_mass_source`, `mass_ratio`: as described above but in source frame
- `mass_1`, `mass_2`: the mass of the primary and secondary object in the detector frame, in solar masses $M_{\odot}$;
- `mass_1_source`, `mass_2_source`: the mass of the primary and secondary object in the source frame, in solar masses $M_{\odot}$;
The Fisher parameters should correspond!
Every time a combination of masses in source frame is passed the additional `redshift` parameter should be passed (overwise an error will be raised!)


[!WARNING]
Including parameters which do not apply for a specific approximants (e.g. giving `'a_1'` 
for `'gwfish_TaylorF2'`) will not currently raise an error, 
the parameters will be simply ignored.

