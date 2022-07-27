## Building the documentation

The HTML documentation may be built from this folder with the command

```bash
make html
```

and the result will be `html` files in the `./build/html` folder,
while a LaTeX version may be built with 

```bash
make latexpdf
```

which will make a PDF file `./build/latex/gwfish.pdf`.

In order for this to work, some packages need to be installed: they 
are all specified as development dependencies in the `pyproject.toml`
for GWFish, and can be installed with `poetry install`.