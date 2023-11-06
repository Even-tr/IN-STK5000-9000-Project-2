import warnings
warnings.simplefilter('always', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import datetime
import time

from sklearn.metrics import accuracy_score
from sklearn import tree
import random

from argparse import ArgumentParser
import os

# Local imports
from helpers import outliers_IQR, outliers_z_score,  handle_outliers, fix_obesity
from helpers import combined_outliers, BMI, fix_polydipsia 

def seed_everything(seed_value=1704):
    """
    Set random seed for reproducibility
    Sets seeds for os, numpy
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value) # set PYTHONHASHSEED env var at fixed value
    random.seed(seed_value) # set fixed seed for python random
    np.random.seed(seed_value) # set fixed seed for numpy



def data_cleaning(dataframe): 
    binary_features = ['Obesity', 'TCep', 'Polydipsia', 'Sudden Weight Loss', 'Weakness',
                    'Polyphagia', 'Genital Thrush', 'Visual Blurring', 'Itching',
                    'Irritability', 'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Alopecia']
    cat_features = ['Race',	'Occupation',	'GP']
    num_features = ['Age',	'Height',	'Weight',	'Temperature',	'Urination']
    one_hot_features = {} # for future use

    # Converts all binary features to lower case
    for f in binary_features:
        dataframe[f] = dataframe[f].str.lower()

    # Converts all binary features to ints, preserving Na-s
    dataframe = dataframe.replace({'yes':1, 'no':0}) 
    dataframe = dataframe.replace({'Positive':1, 'Negative':0})

    # Duplicates

    # We delete them. We assume they are caused by an error in the data collection, and it's unlikely that there are two correct instances with the exact same values.  
    dataframe = dataframe.drop_duplicates(keep='first')

    # Meters to centimeters
    condition = dataframe['Height'] < 100
    dataframe.loc[condition, ['Height']] = dataframe.loc[condition, ['Height']].mul(100)

    # Dropping females and Non-Whites
    # Note missing values will also be dropped
    dataframe = dataframe[dataframe['Gender'] == 'Male']
    dataframe = dataframe[dataframe['Race'] == 'White']


    # Missing categorical data
    # If we don't fill missing categorical data now, we run the risk that either the train or the test set don't contain any NA-s.
    # This can cause a difference in columns after one hot encoding and lead to a crash.
    # we want to fill early s.t. there are no Na-s beyond this point
    dataframe[cat_features + ['Gender']] = dataframe[cat_features+ ['Gender']].fillna('MISSING')

    return dataframe


def outliers(train, test, num_features):
    """
    Identify bounds for what is an outlier.
    by:
    - domain knowledge
    - statistical methods:
        - Z score
        - IQR (with: level 1.5)
        - Most conservative (in reespect to not deem something an outlier chosen) 
    """

    # Identify Boundaries

    # Init with domain knowledge
    train_outlier_bounds = pd.DataFrame(
        {'Lower': [16, 110, 30, np.nan, np.nan], # age: 16, height: 110, weight: 30
        'Upper': [120, 240, 200, np.nan, np.nan]}, # age, height, weight
        index=num_features
    )

    # Statistical IQR and Z-score
    for f in ['Temperature', 'Urination']:
        l_IQR, u_IQR = outliers_IQR(train, f)
        l_Z, u_Z = outliers_z_score(train, f)
        train_outlier_bounds.loc[f] = [max(min(l_IQR, l_Z),0), max(u_IQR, u_Z)]
 
    # remove the identified outliers to missing values
    train = handle_outliers(train, train_outlier_bounds)    
    test = handle_outliers(test, train_outlier_bounds)

    # Combined outliers
    # ;ust be handled after fixing the individual ones, otherwise the same ones would be discovered
    zs_train = combined_outliers(train[num_features], num_features)
    # Handle   
    train = train[zs_train < 4] 
    zs_test = combined_outliers(test, num_features, test.copy())
    test = test[zs_test < 4]
    

    return train, test



def handle_missing_data(train, test):
    """
    Handle missing data
    """
    # Derived Features
    # skrive litt om hvorfor vi har valgt disse her
    train = fix_obesity(train)
    test = fix_obesity(test)

    train = fix_polydipsia(train)
    test = fix_polydipsia(test)

    # Missing Binaries
    train[binary_features] = train[binary_features].fillna(0) # set to false 
    test[binary_features] = test[binary_features].fillna(0) # set to false 

    # Missing numeric
    # Fill Na-s with mean. 
    train[num_features] = train[num_features].fillna(train[num_features].mean())
    # We fill the test data with the mean of the train data, making the test set indepentent of each others
    test[num_features] = test[num_features].fillna(train[num_features].mean())

    # Testing for remaining Na-s
    assert train.isna().sum().sum() == 0, f'train data still containts {train.isna().sum().sum()} Na-s'
    assert test.isna().sum().sum() == 0, f'test data still containts {test.isna().sum().sum()} Na-s'
    return train, test


def create_dummies(train, test):
    """
    Create dummy variables
    """
    gender_dummies_train = pd.get_dummies(train['Gender'], prefix='gender')
    train = train.join(gender_dummies_train)

    gender_dummies_test = pd.get_dummies(test['Gender'], prefix='gender')

    test = test.join(gender_dummies_test)

    one_hot_features['Gender'] = list(set(list(gender_dummies_train.columns) + list(gender_dummies_test.columns)))

    # Add one hot variable column to train/test if it exists for one, but not the other adn fill with 0
    for i in list(gender_dummies_train):
        if i not in gender_dummies_test:
            test[i] = 0

    for i in list(gender_dummies_test):
        if i not in gender_dummies_train:
            train[i] = 0

    def checkEqualColumns(L1, L2):
        return len(L1) == len(L2) and sorted(L1) == sorted(L2)

    assert checkEqualColumns(train.columns, test.columns), f'train and test set doesent have the same features'

    for i in zip(sorted(train.columns), sorted(test.columns)):
        assert i[0] == i[1], 'Train and test columns do not match'


    train['BMI'] = BMI(train['Weight'],  train['Height'])
    test['BMI'] = BMI(test['Weight'],  test['Height'])


    assert len(train.index) > 10

    # Some sanity checks
    assert train.isna().sum().sum() == 0, 'No Na-s should be present after handling. They must have been introduced'

    return train, test


if __name__ == "__main__": 
    """
    Run through the script
    """   
    
    parser = ArgumentParser()
    parser.add_argument("--description", action="store", type=str, default="Project-2")
    parser.add_argument("--infile", action="store", type=str, default='diabetes.csv')
    parser.add_argument("--seed", action="store", type=int, default=2023)
    parser.add_argument("--outfile", action="store", type=str, default='outfile.txt') 
    parser.add_argument("--remove-features", action="store", type=str, default='none') # Not used will be removed
    args = parser.parse_args()

    print("\nDescription :",args.description, "\n")

    # writing essential info to terminal / outfile"
    print("\n\n==========================================================\n")
    print(str(datetime.datetime.now()))
    print("\n\nArgs: \n")
    for arg in vars(args):
        print(' {} {}'.format(arg, getattr(args, arg) or ''))
    
    # set all kinds of seeds for reproducibility
    seed_everything(args.seed)

    #open and write to file
    outfile = open(args.outfile, 'a')
    outfile.write("\n\n==========================================================\n")
    outfile.write(str(datetime.datetime.now()))
    start_time = time.time()


    try:
        infile = args.infile # reac command line argument

    except IndexError:
        # default file is diabetes.csv
        infile = 'diabetes.csv'

    try:
        diabetes = pd.read_csv(infile)

    # Handle imporper filenames
    except FileNotFoundError: 
        print(f'\n\nERROR: file {infile} was not found')
        quit()

    outfile.write(str(infile))
    print(f'Reading: {infile}')


    binary_features = ['Obesity', 'TCep', 'Polydipsia', 'Sudden Weight Loss', 'Weakness',
                    'Polyphagia', 'Genital Thrush', 'Visual Blurring', 'Itching',
                    'Irritability', 'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Alopecia']
    cat_features = ['Race',	'Occupation',	'GP']
    num_features = ['Age',	'Height',	'Weight',	'Temperature',	'Urination']
    one_hot_features = {} # for future use


    # ######## INITIAL DATA ANALYSIS ########
    """
    # Code used for answering questions in this section
    """

    # Data Cleaning

    clean_diabetes = data_cleaning(diabetes)

    # Train - Test split
    train_proportion = 0.8
    train_idx = np.random.choice(clean_diabetes.index, int(train_proportion*len(clean_diabetes.index)), replace=False)
    train = clean_diabetes.loc[train_idx]

    test = clean_diabetes.drop(train_idx)
    assert len(clean_diabetes.index) == len(train.index) + len(test.index)

    # Handle Outliers 
    train, test = outliers(train, test, num_features)

    # Handle Missing data 
    train, test = handle_missing_data(train, test)
    
    # Create dummy variables 
    train, test = create_dummies(train, test)

    # Feature selection
    selected_features = num_features + binary_features 
    selected_features.remove('Urination')
    selected_features.remove('Temperature')
    selected_features.remove('Obesity')
    selected_features.remove('TCep')

    X_train = train[selected_features]
    y_train = train['Diabetes']

    # Test set
    X_test = test[selected_features]
    y_test = test['Diabetes']

    for index in X_train.dtypes.keys():
        dtype = X_train.dtypes[index]
        assert dtype == 'float64' or dtype == 'int64' or dtype == 'uint8', f"feature '{index}' is not of type float or int but {dtype}"

    # Model training
    clf = tree.DecisionTreeClassifier(max_depth=7)
    clf_full_tree = clf.fit(X_train, y_train)

    # predict
    y_test_pred = clf_full_tree.predict(X_test)

    # Some Pruning
    # Code borrowed from https://www.kaggle.com/code/arunmohan003/pruning-decision-trees-tutorial

    # Alpha 0.02 seems like a good trade off between size and complexity
    # as indicated by pruned_tree_complexity.png
    alpha = 0.02
    classes = ['Negative', 'Positive']
    clf_pruned_tree = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
    clf_pruned_tree.fit(X_train,y_train)
    y_train_pred = clf_pruned_tree.predict(X_train)
    y_test_pred = clf_pruned_tree.predict(X_test)

    print('RESULTS')
    print('Pruned tree')
    print(f'Test Accuarcy {accuracy_score(y_test_pred,y_test)}')
    print(f'\nFinished in {time.time() - start_time:.2f} seconds')

    tree.plot_tree(clf_pruned_tree, feature_names=selected_features, class_names=classes, filled=True)