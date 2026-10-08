# Local install instructions

The course uses Python 3.11 or above, scikit-learn, pandas, matplotlib, skrub,
skore and [TabICL](https://github.com/soda-inria/tabicl).

TabICL depends on PyTorch, and the first time a TabICL model is fitted it
downloads its pretrained weights (about 110 MB) to a local cache.

## Install Miniforge

**This step is only necessary if you don't have conda installed already**:
download and run the [Miniforge](https://conda-forge.org/download/) installer
for your operating system.

## Create the conda environment

```sh
# Clone this repo
git clone https://github.com/probabl-ai/TabICL-course.git
cd TabICL-course
# Create a conda environment with the required packages for this course
conda env create -f environment.yml
```

Alternatively, if you prefer pip, install the packages listed in
`requirements.txt` in a virtual environment:

```sh
pip install -r requirements.txt
```

## Check your install

To make sure you have all the necessary packages installed, we **strongly
recommend** you to execute the `check_env.py` script located at the root of
this repository:

```sh
conda activate tabicl-course
python check_env.py
```

Make sure that there is no `FAIL` in the output.

## Run Jupyter notebooks locally

```sh
conda activate tabicl-course
jupyter lab
```

All the Jupyter notebooks are located in the `notebooks` folder.
