
# # Simple pipeline to add differential privacy to a data set

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

# Pipeline imports
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


_binary_features = ['Obesity', 'TCep', 'Polydipsia', 'Sudden Weight Loss', 'Weakness',
                'Polyphagia', 'Genital Thrush', 'Visual Blurring', 'Itching',
                'Irritability', 'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Alopecia', 'Gender']
_cat_features = ['Race',	'Occupation',	'GP']
_num_features = ['Age',	'Height',	'Weight',	'Temperature',	'Urination']

_target = 'Diabetes'

def randomize_binary(a, theta):
    """
    Accepts a vector of binary values and add randomized noise, parameterized by theta = probability of answering truthfully.
    """
    coins = np.random.choice([True, False], p=(theta, (1-theta)), size=a.shape)
    noise = np.random.choice(['no', 'yes'], size=a.shape)
    response = np.array(a)
    response[~coins] = noise[~coins]
    return response

def randomize_categorical(a, theta):
    """
    placeholder function for adding privacy noise for categorical columns
    """
    return a

def randomize_numerical(a, theta):
    """
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, _num_features),
            ("cat", cat_transformer, _cat_features),
            ('binary', binary_transformer, _binary_features)
        ],
        verbose_feature_names_out= True,
        remainder='drop'                # drop untouched features since after this step, as it is the last preprocessing one
    ).set_output(transform="pandas")    # Keep data frame format
    
    return preprocessor

def anonymize_data(infile, theta, outfile = None):
    """
    Reads an infile and adds column wise differential privacy with theta probability of telling the truth.

    Target column, diabetes, is not randomized. TODO: shall we randomize it?

    Optionally, it also saves the data frame in the outfile location.
    """

    # read data
    indata = pd.read_csv(infile)
    y = indata[_target]
    X = indata.drop(columns=(_target))

    # make pipeline
    preprocessor = make_privacy_pipeline(theta=theta)

    # run pipelein
    out_data = preprocessor.fit_transform(X,y)

    # reformat to same match infile
    out_data.columns = _num_features + _cat_features + _binary_features
    out_data['Gender'] = out_data['Gender'].replace({'yes':'Male', 'no':'Female', 'Yes':'Male', 'No':'Female'}) 
    out_data['Diabetes'] = y
    out_data = out_data.reindex(columns=indata.columns)

    # save data 
    if outfile is not None:
        out_data.to_csv(outfile)

    return out_data

if __name__ == '__main__':
    assert calculate_epsilon(0.5) == np.log(3) # result from lecture notes

    diabetes = pd.read_csv('diabetes.csv')

    # with probability of telling the truth equal to one, there should be no anonymization
    pd.testing.assert_frame_equal(diabetes, anonymize_data('diabetes.csv', 1), check_dtype=False)

    # with only a slight probability of not answering truthfully, there should be some anonymization
    # Note: there is a probability greater than zero that this test fails even when it works as expected
    try:
        pd.testing.assert_frame_equal(diabetes, anonymize_data('diabetes.csv', 0.99), check_dtype=False)
    except AssertionError:
        pass
    else:
        raise AssertionError
