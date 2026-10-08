# Contributing

The source files, which should be modified, are in the `python_scripts`
directory. The notebooks in `notebooks` are generated from these Python files
with [Jupytext](https://jupytext.readthedocs.io/): do not edit them by hand.

Set up your environment with the `environment-dev.yml` file:

```sh
conda env create -f environment-dev.yml
conda activate tabicl-course-dev
```

## Recommended workflow: Visual Studio Code

- Download [Visual Studio Code](https://code.visualstudio.com/)
- Install the [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter).
  This will allow you to run Python files in a cell by cell fashion

Other editors (e.g. PyCharm) may allow a similar workflow: the only feature
you need is to be able to run a `.py` file in a cell-by-cell fashion with
`# %%` cell markers.

## Updating the notebooks

`build_tools/convert-python-script-to-notebook.py` converts a script to a
notebook with Jupytext, and also renders MyST admonitions (e.g. ```` ```{tip} ````)
as HTML, since Jupyter does not render them natively.

To update all the notebooks:

```sh
for f in python_scripts/*.py; do
    python build_tools/convert-python-script-to-notebook.py "$f" "notebooks/$(basename "${f%.py}").ipynb"
done
```

To update a single notebook:

```sh
python build_tools/convert-python-script-to-notebook.py python_scripts/TabICL_intro.py notebooks/TabICL_intro.ipynb
```
