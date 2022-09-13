# Choosing an approximant

In order to change the approximant used when running `CBC_Simulation.py`,
modify its source code changing the variable `waveform_model` to the desired
value. 

Valid options are the following:
- `'gwfish_TaylorF2'`, a simple and analytic post-Newtonian approximant;
- `'gwfish_IMRPhenomD'`, a tuned high-order approximant for binary black holes;
- `'lalsim_*'`, where `*` is the name of any frequency-domain approximant available in 
    LALSimulation (see [here](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_inspiral__h.html#gab955e4603c588fe19b39e47870a7b69c) for some options)
    - examples include `'lalsim_IMRPhenomXPHM'`, `'lalsim_IMRPhenomD'`.