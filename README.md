### Prerequisites 
We provide several different ways to install the required packages:

## Virtual environments
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

## Global environment
If however you want to install dependencies directly, please use the following command, but be advised that you might encounter compatibility errors.

```bash
pip install -r requirements.txt
```
## Environment in conda:
```bash
conda env create -f environment.yml
```


### HOW TO RUN

This project is build using a makefile, which is a convenient interface for more complex command line use. Linux and Mac should have it preinstalled, but Windows user might need to install it themselves (see https://gnuwin32.sourceforge.net/packages/make.htm for reference - and remember to add it to the path!).

To use it, simply write 'make' followed by the predefined run you want to use. Our main experiment is run with:

```bash
make all
```

Which first runs our baseline implementation, then our new implementation with N_SAMPLES=100. (i.e. 100 bootstrap samples for the score ranges). It then reruns our new implementation with two levels of anonymization. In the first one, the probability of answering truthfully is 0.5, yielding ln(3) privacy. In the second experiment, this is much higher at 0.95.

For general usage, some examples presented below showing the interface.

```bash
make run INFILE=<your_file_name>
```

The makefile accepts the following arguments:

```
INFILE      = diabetes.csv
OUTFILE     = anon.csv
THETA       = 0.95
N_SAMPLES   = 100
ANON_SEED   = -1 
FIGFOLDER   = ./figs/
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

#### Command line arguments
Our experiments are defined by specifying command line arguments. These arguments can also be accessed when using the makefile, and doing it with this abstraction allows for convenient key word argument usage.

```bash
main.py <infile> <bootstrap samples> <figfolder> <theta>
```
The first argument is the dataset which is analyzed. The second is the number of bootstrap samples (we used 100). The third is the destination folder for the generated figures - specifying a unique folder for each run avoids overwriting. The final argument is simply the theta used when privatizing. It only adds this information to the figures, and does not impact the runtime. If this is unspecified, this information will be omitted.

```bash
privacyPipe.py <infile> <outfile> <theta> <seed>
```
The first argument is the path to the data set to be anonymized. The second is the path to the new data set which is created. The third argument specifies the probability of answering truthfully (theta) - the higher the theta the less anonymization. The final argument is the seed used for anonymization. If this is unspecified, a truly random (and therefore non-reproducible) data set is generated.
# IN-STK5000-9000-Project-2
