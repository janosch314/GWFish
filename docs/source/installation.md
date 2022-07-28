# Installing GWFish

An older version of GWFish is [available on pypi](https://pypi.org/project/GWFish/),
but don't use it. Instead, clone the repo.

Move to a clean folder, and run:

```bash
git clone https://github.com/janosch314/GWFish
cd GWFish
```

Now you should have a shell prompt ready in the `GWFish` folder, 
and by running `ls` you should see a folder again called `GWFish`, 
as well as three scripts called `CBC_Simulation.py`, `CBC_Population.py` 
and `CBC_Background.py`.

You can then install `GWFish` with:

```bash
pip install .
```

## Extra waveform packages

The procedure above will not automatically install `lalsuite` - it's a heavy dependency; 
on the other hand running

```bash
pip install .[waveforms]
```

will also install `lalsuite`.

This dependency is separated from the rest since it's not strictly necessary in order for `GWFish` to work:
it provides extra waveform [approximants](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_inspiral__h.html#gab955e4603c588fe19b39e47870a7b69c).

Within GWFish itself, a simple Post-Newtonian approximant (`TaylorF2`) and a binary-black-hole 
Inspiral-Merger-Ringdown phenomenological approximant (`IMRPhenomD`) are implemented.

```{todo} 
In the near future, also the machine-learning BNS surrogate [`mlgw_bns`](https://pypi.org/project/mlgw-bns/)
will be implemented and included as an optional dependency in the `waveforms` group.
```

## Development installation

In order to work on the development of the project,
[install `poetry`](https://python-poetry.org/docs/master/#installing-with-the-official-installer).

Then, run the following commands:

```bash
poetry shell
poetry install
```

The first will create a virtual environment specifically for this project,
while the second will install all the dependencies, including the development ones.