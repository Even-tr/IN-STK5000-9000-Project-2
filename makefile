INFILE ?= diabetes.csv
OUTFILE ?= anon.csv
THETA ?= 0.95
N_SAMPLES = 20

# Simple analysis with normal data
all:
	python ./pipeline.py

normal:
	python ./pipeline.py $(INFILE) $(N_SAMPLES)

anon:
	python ./privacyPipe.py $(INFILE) $(OUTFILE) $(THETA)
	python ./pipeline.py $(OUTFILE) $(N_SAMPLES)


analysis:
	make normal
	make anon