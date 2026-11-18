PYTHON_SCRIPTS_DIR = python_scripts
NOTEBOOKS_DIR = notebooks
# Python with the dependencies of environment-dev.yml, e.g.
# make PYTHON=~/miniforge3/envs/tabicl-course-dev/bin/python
PYTHON ?= python

NOTEBOOK_FILES = $(patsubst $(PYTHON_SCRIPTS_DIR)/%.py,$(NOTEBOOKS_DIR)/%.ipynb,$(wildcard $(PYTHON_SCRIPTS_DIR)/*.py))
EXERCISE_FILES = $(subst _sol_,_ex_,$(wildcard $(PYTHON_SCRIPTS_DIR)/*_sol_*.py))

all: $(NOTEBOOKS_DIR) prompts

.PHONY: all $(NOTEBOOKS_DIR) exercises prompts

$(NOTEBOOKS_DIR): $(NOTEBOOK_FILES) $(NOTEBOOKS_DIR)/matplotlibrc

$(NOTEBOOKS_DIR)/%.ipynb: $(PYTHON_SCRIPTS_DIR)/%.py
	$(PYTHON) build_tools/convert-python-script-to-notebook.py $< $@

$(NOTEBOOKS_DIR)/matplotlibrc: $(PYTHON_SCRIPTS_DIR)/matplotlibrc
	cp $< $@

# Each exercise is generated from its solution, so edit the solution only.
exercises: $(EXERCISE_FILES)

$(PYTHON_SCRIPTS_DIR)/TabICL_ex_%.py: $(PYTHON_SCRIPTS_DIR)/TabICL_sol_%.py
	$(PYTHON) build_tools/generate-exercise-from-solution.py $@

# Prompts of the AI tutor deployed with the notebooks in JupyterLite.
prompts: $(NOTEBOOKS_DIR)/global-prompt.json

$(NOTEBOOKS_DIR)/global-prompt.json: build_tools/generate-agent-prompts.py | $(NOTEBOOK_FILES)
	$(PYTHON) build_tools/generate-agent-prompts.py
