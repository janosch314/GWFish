# Temporal truncation

## Mission lifetime constraints

<!-- TODO add a figure -->

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

Non-{term}`CBC` sources do not respect these considerations, but at the moment `GWFish` 
does not offer strong support for them.

## `max_frequency_cutoff` details

This [parameter](parameter-definitions-units) allows one to truncate the waveform at a specific upper frequency.
It is integrated with the mission lifetime constraint, i.e. the "coalescence time"
is adapted to be the time for which the given frequency is reached.

## The `redefine_tf_vectors` parameter

Certain GWFish functions (see [here](#api-reference)) offer the boolean parameter `redefine_tf_vectors`.
In the default configuration this is `False`, which means that in the computation of all relevant 
integrals the frequency grid is fixed to be the one from the detector definition (for examples see [the list of included detectors](#included-detectors)).

For the typical {term}`CBC` sources seen by ground-based detectors,