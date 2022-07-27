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

What should then work is

```bash
pip install .
```

```{error} This seems to be buggy now! 
```