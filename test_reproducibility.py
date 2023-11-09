# tests reproducbility when anonymizing data
import subprocess
import pandas as pd

outfile1 = 'test_anon1.csv'
outfile2 = 'test_anon2.csv'
theta= 0.9


def test_reproducible_anonymization():
    """
    Tests that the anonymization is reproducible, given a seed.

    It runs the privacyPipe.py script twice and saves the two anonymized 
    data sets, then asserts that they are equal.
    """

    seed = 2023

    # generate first data set
    command = f"make anonymize OUTFILE={outfile1} THETA={theta} ANON_SEED={seed}"
    subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, text=True)

    # generate second data set
    command = f"make anonymize OUTFILE={outfile2} THETA={theta} ANON_SEED={seed}"
    subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, text=True)


    df1 = pd.read_csv(outfile1)
    df2 = pd.read_csv(outfile2)

    pd.testing.assert_frame_equal(df1, df2, check_dtype=False)


def test_non_reproducible_anonymization():
    """
    Tests that the anonymization is random, when not supplying a seed.

    It runs the privacyPipe.py script twice and saves the two anonymized 
    data sets, then asserts that they are equal.
    """

    # generate first data set
    command = f"make anonymize OUTFILE={outfile1} THETA={theta}"
    subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, text=True)

    # generate second data set
    command = f"make anonymize OUTFILE={outfile2} THETA={theta}"
    subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, text=True)


    df1 = pd.read_csv(outfile1)
    df2 = pd.read_csv(outfile2)

    try:
        # we expect, and intend, this test to fail
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False)
    except AssertionError: 
        pass
    else:
         # Raise assertion error if it does not fail
         assert False, 'Truly randomized data has been reproduced'


def test_pipe_reproducibilty():
    """
    Tests that two separate runs of the pipeline yeilds the same result.

    It calls a subprocess and compares the standard out for the same command twice.
    """
    n_samples = 1 # select a low number of samples to save time
    command = f"make run N_SAMPLES={n_samples} FIGFOLDER=./testfigs/"

    # generate first output
    res1 = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, text=True)

    # generate second output
    res2 = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, text=True)

    assert res1.stdout == res2.stdout, 'Non reproducible pipeline'

if __name__ == '__main__':
    test_pipe_reproducibilty()