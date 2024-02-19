# Installing GWFish

An older version of GWFish is [available on pypi](https://pypi.org/project/GWFish/),
but don't use it --- it's not up to date yet. Instead, clone the repo.

Move to a clean folder, and run:

```bash
git clone https://github.com/janosch314/GWFish
cd GWFish
```
You can then install `GWFish` with:

```bash
pip install .
```

Now you should be able to use it from anywhere in your system.

## Extra waveform packages

The procedure above will not automatically install `lalsuite` - it's a heavy dependency; 
on the other hand running

```bash
pip install .[waveforms]
```

will also install `lalsuite`.

This dependency is separated from the rest since it's not strictly necessary in order for `GWFish` to work:
it provides extra waveform [approximants](https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/group___l_a_l_sim_inspiral__h.html#gab955e4603c588fe19b39e47870a7b69c), see also [the approximants reference](../how-to/choosing_an_approximant.md).

Within GWFish itself, a simple Post-Newtonian approximant (`TaylorF2`) and a binary-black-hole 
Inspiral-Merger-Ringdown phenomenological approximant (`IMRPhenomD`) are implemented.

## Development installation

In order to work on the development of the project,
[install `poetry`](https://python-poetry.org/docs/master/#installing-with-the-official-installer).

Then, run the following commands:

```bash
poetry shell
poetry install --with dev --with docs
```

The first will create a virtual environment specifically for this project,
while the second will install all the dependencies, including the development ones
and the ones required to build the documentation.

### Running the tests

The test suite can be run from the main `GWFish` folder with the command

```bash
pytest
```

This will run all tests - most are in the `tests` folder, but some also 
check the documentation and are therefore in the `docs` folder, or in the
docstrings of the functions themselves in the `GWFish` folder.

### Building the documentation

The documentation can be built from the `docs` folder by running 
the command 

```bash
make html
```