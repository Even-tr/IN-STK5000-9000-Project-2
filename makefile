INFILE ?= diabetes.csv
OUTFILE ?= anon.csv
THETA ?= 0.95
N_SAMPLES = 20
ANON_SEED = -1 # default value means true randomization

# Simple analysis with normal data
all:
	python ./pipeline.py

# Analysis with specified infile and bootstrap samples
run:
	python ./pipeline.py $(INFILE) $(N_SAMPLES)

# Anonymized Analysis with specified infile and bootstrap samples
anonamized_run: anonymize
	python ./pipeline.py $(OUTFILE) $(N_SAMPLES)

# Anonymizes the data
anonymize:
	python ./privacyPipe.py $(INFILE) $(OUTFILE) $(THETA) $(ANON_SEED)
