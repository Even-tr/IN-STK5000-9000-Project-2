### Prerequisites
#### Virtual environments
We recommend using virtual environments, and made a simple scripts which creates and install the required packages. This guide is made for Unix based systems.
For windows, consider using WSL, or consult the documentation: https://docs.python.org/3/library/venv.html

To create the environment, use this command (Unix):

```bash
make venv
```

(windows)
```cmd
make venv_windows
```
and activate it with this command (Unix):

```bash
source ./IN-STK5000/bin/activate
```

(windows)
```cmd
.\IN-STK5000\Scripts\activate
```
The virtual environment can be deactivated using this command:

```bash
deactivate
```
And finally, the environment can be cleaned using this command (Unix):

```bash
make clean_venv
```

Or delete the folder manually.

#### Global environment
If however you want to install dependencies directly, please use the following command, but be advised that you might encounter compatibility errors.

```bash
pip install -r requirements.txt
```
or create environment:
```bash
conda env create -f environment.yml
```


### HOW TO RUN

This project is build using a makefile, which is a convenient interface for more complex command line use. Linux and Mac should have it preinstalled, but Windows user might need to install it themselves (see https://gnuwin32.sourceforge.net/packages/make.htm for reference - and remember to add it to the path!).

To use it, simply write 'make' followed by the predefined run you want to use. For instance,

```bash
make all
```

runs the first predefined command in the make file, which simply runs the analysis pipeline with default arguments. If you want to run the analysis on a different file, please write 

```bash
make run INFILE=<your_file_name>
```

The makefile accepts the following arguments:

```
INFILE = diabetes.csv
OUTFILE = anon.csv
THETA = 0.95
N_SAMPLES = 20
ANON_SEED = -1 
FIGFOLDER = ./figs/
```

Where the right-hand side, displaying the default values, can be replaced with the desired parameters.

#### Some example runs
1. Run with a different file
```bash
make run INFILE=anon.csv
```
2. Anonymize the diabetes dataset with probability of answering truthfully = 0.9
```bash
make anonymize OUTFILE=anon.csv THETA=0.9
```

3. Anonymize and then analyze
```bash
make anonymized_run OUTFILE=anon.csv THETA=0.9
```

4. Anonymize, but with a known seed s.t. it can be reproduced, and then analyze
```bash
make anonymized_run OUTFILE=anon.csv THETA=0.9 ANON_SEED=2023
```


# LENKER SOM SLETTES VED INNLEVERING 
https://docs.google.com/presentation/d/1U_PAywkxoO0AGcN5XDDShKzrxjGgmvw63qKXIF6WFFY/edit?usp=sharing

## Liten todo som slettes v innlevering 
## hva som skal gjøres i koden nå: 
### cleaning: 
~~- Normalisation~~
 ~~- convert binary and categorical features to lower case
 ~~-convert binary features to ints
- M -> cm 
### delete duplicates 
### One hot encoding -> other categori
- outliers pipileine
~~missing data in i pipeline
~~feature selection : legg til kode for -remove-features

~~legge til privacy: mynt og kron på categorical, laplace på numeriske. EGET SEED FOR PRIVACY. alfa/ theta experimenter (som forelesning 16 okt). 
~~Cross validation for Alfa

~~Metrics: acc, prec, rec, f1 (si noe om privacy også 

~~bootstrapping eller cross-v: rapportering på range

# IN-STK5000-9000-Project-2
