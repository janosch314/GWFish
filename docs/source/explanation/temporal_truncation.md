# Temporal truncation

## Mission lifetime constraints

In the detector projection computation for space- and Moon-based detectors,
a truncation is applied at low frequencies: specifically, if the coalescence
happens at time $t_c$, the waveform is set to zero for all times $t$ such that 

$$ t < t_c - t_m,
$$

where $t_m$ is the mission lifetime.
This is an optimistic choice, since it amounts to assuming that the system at hand 
will merge precisely at the end of the mission's lifetime. 

This correction is not required for Earth-based detectors, since they are 
necessarily constrained to $f \gtrsim 1 \text{Hz}$, and even the lightest compact 
objects will not take more than a few days to merge from those frequencies, which
is much shorter than any sensible mission duration.

Non-CBC sources do not respect these considerations, but at the moment `GWFish` 
does not offer strong support for them.

## `max_frequency` details

This [parameter](parameter-definitions-units) allows one to truncate the waveform at a specific upper frequency.
It is integrated with the mission lifetime constraint, i.e. the "coalescence time"
is adapted to be the time for which the given frequency is reached.

```{warning}
Very low values for this parameter for space- or Moon-based detectors run the risk 
of having a completely zero signal vector. 

Specifically, it can happen that the temporal difference $t(f_0)$ and $t(f_0 - \Delta f)$
is larger than the mission lifetime. 
This will manifest with an SNR being exactly equal to zero, and it can be
solved by making $\Delta f$ smaller; that will make the computation take longer,
but that can be somewhat ameliorated by temporarily lowering the maximum frequency
for the frequency arrays (i.e. the `fmax` parameter in the detector definition).
```