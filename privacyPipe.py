
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

# Pipeline imports
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import sys

# Read arguments from command line
try:
    infile = sys.argv[1]
    outfile = sys.argv[2]
    theta = float(sys.argv[3])
except IndexError:
    infile = "diabetes.csv"
    outfile = "anon.csv"
    theta = 0.95
    print(f"Default arguments used: {infile}, {outfile}, {theta}")

# Read seed from command line
try:
    seed = int(sys.argv[4])
    if seed == -1:
        seed = np.random.randint(2**16 - 1) # sample random seed from 0 to max int
        print('Non Reproducible seed used')
except IndexError:
    seed = np.random.randint(2**16 - 1)
    print('Non Reproducible seed used')

# Set seed
np.random.seed(seed)

# Define features
binary_features = ['Obesity', 'TCep', 'Polydipsia', 'Sudden Weight Loss', 'Weakness',
                'Polyphagia', 'Genital Thrush', 'Visual Blurring', 'Itching',
                'Irritability', 'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Alopecia', 'Gender']
cat_features = ['Race',	'Occupation',	'GP']
num_features = ['Age',	'Height',	'Weight',	'Temperature',	'Urination']


target = 'Diabetes'


def randomize_binary(a, theta):
    """
     Inject random noise into a vector of binary responses based on a given probability (theta). 
     The probability theta represents the chance of a truthful response. 

     Parameters:
        a (array-like): vector of binary responses (no, yes / 1, 0)
        theta (float): probability of answering truthfully

    Returns:
        array-like: vector of randomized responses
    """
    # Generates a set of random boolean values with a probability theta of being True. 
    coins = np.random.choice([True, False], p=(theta, (1-theta)), size=a.shape) 
    #  Generates a set of random strings ('no', 'yes', 'No', 'Yes') to match original data format 
    noise = np.random.choice(['no', 'yes', 'No', 'Yes'], size=a.shape)
    # creates a copy of the original data 
    response = np.array(a)
    # replaces the values with noise where the coin is False
    response[~coins] = noise[~coins]
    return response

def randomize_categorical(a, theta):
    """
    TO DO IN FUTURE
    placeholder function for adding privacy noise for categorical columns
    """
    return a

def randomize_numerical(a, theta):
    """
    TO DO IN FUTURE
    placeholder function for adding privacy noise for numerical columns
    """
    return a

def calculate_epsilon(theta):
    """
    Calculates the amount of differential privacy for a single column
    given a biased coin theta. The secondary coin, used for random answering, is 
    assumed to be fair.

    It is calculated by the equation

        P(y|T=y)/P(n|T=y) <= exp(epsilon)

    with 
        P(y|T=y) = theta + (1-theta)*0.5
    and
        P(n|T=y) = (1-theta)*0.5

    Se biased_coin_privacy.md for details.
    """

    return np.log( (theta +(1-theta)*0.5)/(((1-theta)*0.5)))

assert calculate_epsilon(0.5) == np.log(3) # result from lecture notes


def make_privacy_pipeline(theta):   
    """
    Takes a probability for answering truthfully and creates a pipeline which adds
    noise accordingly.

    ----
    returns:
        Pipeline
    """    
    binary_transformer = Pipeline(
        steps=[
            ('randomize', FunctionTransformer(randomize_binary, kw_args={'theta': theta})),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ('randomize', FunctionTransformer(randomize_categorical, kw_args={'theta': theta})),
        ]
    )

    num_transformer = Pipeline(
        steps=[
            ('randomize', FunctionTransformer(randomize_numerical, kw_args={'theta': theta})),
        ]
    )


    # Combines steps to a final pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
            ('binary', binary_transformer, binary_features)
        ],
        verbose_feature_names_out= True,
        remainder='drop'                # drop untouched features since after this step, as it is the last preprocessing one
    ).set_output(transform="pandas")    # Keep data frame format
    
    return preprocessor

def anonymize_data(infile, theta, outfile = None):

    # read data
    indata = pd.read_csv(infile)
    y = indata[target]
    X = indata.drop(columns=(target))

    # make pipeline
    preprocessor = make_privacy_pipeline(theta=theta)

    # run pipeline to add noise
    out_data = preprocessor.fit_transform(X,y)

    # reformat to same match infile
    out_data.columns = num_features + cat_features + binary_features
    out_data['Gender'] = out_data['Gender'].replace({'yes':'Male', 'no':'Female', 'Yes':'Male', 'No':'Female'}) 
    out_data['Diabetes'] = y
    out_data = out_data.reindex(columns=indata.columns)

    # save data 
    if outfile is not None:
        out_data.to_csv(outfile)

    return out_data


def test_random():
    """Testing if data is randomized by setting theta to 0.95 """

    out_data = anonymize_data('diabetes.csv', 0.95) # set theta to 0.95
    diabetes = pd.read_csv('diabetes.csv') # read original data
    try:
        pd.testing.assert_frame_equal(diabetes, out_data, check_dtype=False) # compare data
    except AssertionError:
        pass
    else:
        raise AssertionError # "data not randomized"
    

def test_equal():
    """Testing if data is not randomized by setting theta to 1.0 """
    out_data = anonymize_data('diabetes.csv', 1)
    diabetes = pd.read_csv('diabetes.csv')
    pd.testing.assert_frame_equal(diabetes, out_data, check_dtype=False)

test_random()
test_equal()

out_data = anonymize_data(infile, theta, outfile=outfile)
