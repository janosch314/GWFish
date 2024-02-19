# Choosing a waveform approximant

The modelling of gravitational waveforms is complex, and no single 
model ("approximant") at the moment captures all relevant physics for a generic
analysis. 
Some models are natively implemented in GWFish, and they can be accessed by using
the waveform class `GWFish.modules.<waveform_name>`, where `waveform_name` is one of the following (explained later):

- `'TaylorF2'`;
- `'IMRPhenomD'`.

Many more models are available by calling LALSimulation, a code developed by the LIGO-Virgo-Kagra collaboration.
It should be automatically installed, as the python package [`lalsuite`](https://pypi.org/project/lalsuite/), together with GWFish.

In order to use these models in GWFish, specify the waveform class `GWFish.modules.LALFD_Waveform`, and then the name of the model, such as `'IMRPhenomD'`.
For a complete list of options, see [here](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_inspiral__h.html#gab955e4603c588fe19b39e47870a7b69c).
Some models that are commonly used in our analyses are:

- `'TaylorF2'`, a simple and analytic post-Newtonian approximant, used when we only care about the low-frequency part of the signal which is unaffected by high-order effects;
- `'IMRPhenomD'`, a tuned high-order approximant for black hole binaries without higher order modes, the _default choice_ in GWFish;
- `'IMRPhenomXPHM'`, a similar model also including higher order modes (especially important for high-mass and off-axis events);
- `'IMRPhenomD_NRTidalv2'`, a model for neutron star binaries including tidal effects.
