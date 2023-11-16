# Defining variables and default values
INFILE ?= diabetes.csv
OUTFILE ?= anon.csv
THETA ?= 0.95
N_SAMPLES ?= 100
ANON_SEED ?= 1984 # -1 means true randomization
FIGFOLDER ?= ./figs/
VENV_NAME ?=IN-STK5000

# Simple analysis with normal data
all: 
	python ./baseline.py
	python ./main.py 		$(INFILE) $(N_SAMPLES) $(FIGFOLDER)
	python ./privacyPipe.py $(INFILE) $(OUTFILE) 0.5 1984
	python ./main.py 		$(OUTFILE) $(N_SAMPLES) ./anonymized_figs_theta05/ 0.5
	python ./privacyPipe.py $(INFILE) $(OUTFILE) 0.95 1984
	python ./main.py 		$(OUTFILE) $(N_SAMPLES) ./anonymized_figs_theta095/ 0.95

baseline:
	python ./baseline.py


# Analysis with specified infile and bootstrap samples
run:
	python ./main.py $(INFILE) $(N_SAMPLES) $(FIGFOLDER)

# Anonymized Analysis with specified infile and bootstrap samples
anonymized_run: anonymize
	python ./main.py $(OUTFILE) $(N_SAMPLES) ./anonymized_figs/ $(THETA)

# Anonymizes the data
anonymize:
	python ./privacyPipe.py $(INFILE) $(OUTFILE) $(THETA) $(ANON_SEED)


# creates a virual enviroment
venv:
	python3 -m venv $(VENV_NAME) && source $(VENV_NAME)/bin/activate && pip3 install -r requirements.txt

venv_windows:
	python3 -m venv $(VENV_NAME) && .\$(VENV_NAME)\Scripts\activate && pip3 install -r requirements.txt


# removes the virtual enviroment
clean_venv:
	rm -rf $(VENV_NAME)

clean_venv_windows:
	rmdir /s /q $(VENV_NAME)

.PHONY: venv clean_venv venv_windows clean_venv_windows
