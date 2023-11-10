# Defining variables and default values
INFILE ?= diabetes.csv
OUTFILE ?= anon.csv
THETA ?= 0.95
N_SAMPLES ?= 20
ANON_SEED ?= -1 # default value means true randomization
FIGFOLDER ?= ./figs/
VENV_NAME ?=IN-STK5000
# Simple analysis with normal data
all:
	python ./pipeline.py 

# Analysis with specified infile and bootstrap samples
run:
	python ./pipeline.py $(INFILE) $(N_SAMPLES) $(FIGFOLDER)

# Anonymized Analysis with specified infile and bootstrap samples
anonymized_run: anonymize
	python ./pipeline.py $(OUTFILE) $(N_SAMPLES) ./anonymized_figs/

# Anonymizes the data
anonymize:
	python ./privacyPipe.py $(INFILE) $(OUTFILE) $(THETA) $(ANON_SEED)


# creates a virual enviroment
venv:
	python3 -m venv $(VENV_NAME) && source $(VENV_NAME)/bin/activate && pip install -r requirements.txt

# removes the virtual enviroment
clean_venv:
	rm -rf $(VENV_NAME)

.PHONY: venv clean_venv