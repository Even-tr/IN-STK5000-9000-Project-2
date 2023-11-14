import warnings
warnings.simplefilter(action='ignore', category=UserWarning)    # ignore pesky warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # with version specified, future warnings is superfluous (although this is bad practice)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn import tree

# Local imports
from pipeline import make_clf, tune_clf, param_grid, binary_features, cat_features, num_features
from score import score, score_names
from visualise import visualise_feature_importance, visualise_results


if __name__ == "__main__":
    #Set seed for reproducibility
    np.random.seed(2023)

    # Read file from command file argument
    try:
        infile = sys.argv[1]
        print(f'Reading {infile}')
    except IndexError:
        infile = "diabetes.csv"
        print(f"Default arguments used: {infile}")

    try:
        n_samples = int(sys.argv[2])
    except IndexError:
        n_samples = 20

    try:
        figfolder = sys.argv[3] # destination folder for figures
    except IndexError:
        figfolder = './figs/'

    try:
        theta = float(sys.argv[4]) # theta used in anynoymzation (only used for printing)
    except IndexError:
        theta = 1

    print(f'\ntheta = {theta}')
    print("----------------------------------------")

    diabetes = pd.read_csv(infile)

    target = 'Diabetes'

    y = diabetes[target] # target variable
    X = diabetes.drop(columns=(target)) # features

    # If our model has worse accuracy than this, simply guessing 'Postive' is a better model

    print(f"Mean diabetes in data set: {(y=='Positive').mean()}") 
    print("----------------------------------------")


    # Single run resutls
    print('Results for single run of model')
    train, test = train_test_split(diabetes, train_size=0.8)
    X_train = train.drop(target, axis=1)
    y_train = train[target]
    X_test = test.drop(target, axis=1)
    y_test = test[target]

    # decision tree classifier is instantiated and trained
    model =  tune_clf(make_clf(tree.DecisionTreeClassifier()), X_train, y_train)['model']
    results = pd.DataFrame()
    results.index=score_names
    results['train'] = score(model, X_train, y_train)
    results['test'] = score(model, X_test, y_test)
    with pd.option_context('display.float_format', '{:0.3f}'.format):
        print(results)



    # Bootstrapping results
    # n_samples = 20 # number of bootstrap samples
    models = [] # list to save the constructed models
    train_scores = np.zeros(shape=(n_samples, len(score_names))) # array to save train scores
    test_scores = np.zeros_like(train_scores)   # array to save test scores


    """
    Discuss this a lot, since we have thought about it so much. 

    Note: we dont correct the bootstrap sample here since it is not clear if the 
    0.628 correction is applicable with zero-one loss (and as a consequence accuracy),
    and in particular, how we correct other performance measures, like Precision and 
    Recall.
    """

    # Bootstrap loop
    print("----------------------------------------")
    print(f'Generating {n_samples} bootstrap samples') 
    print("Training and tuning models for each sample")

    for i in tqdm(range(n_samples), desc="Bootstrapping", unit="sample"):


    #for i in range(n_samples):
        #print(f'{i/n_samples*100:3.0f}% done')

        # random sampling with replacement.
        idx = np.random.choice(range(len(diabetes.index)), size=len(diabetes.index), replace=True) 
        assert len(set(idx)) < len(idx) # testing for repeated samples

        # split into training and test sets
        train = diabetes.iloc[idx]
        test = diabetes[~diabetes.index.isin(idx)] 

        # testing for complete split
        assert len(test.index) + len(set(idx)) == len(diabetes.index), f"{len(test.index)}, {len(set(idx))}, {len(diabetes.index)}  "
        
        #Resetting index to avoid errors
        train.reset_index(inplace=True)
        test.reset_index(inplace=True)

        # Splitting into X and y
        X_train = train.drop(target, axis=1)
        y_train = train[target]
        X_test = test.drop(target, axis=1)
        y_test = test[target]

        # decision tree classifier is instantiated and trained
        model =  tune_clf(make_clf(tree.DecisionTreeClassifier()), X_train, y_train)['model']

        # Calculate models performance on train and test set
        train_scores[i] = score(model, X_train, y_train)
        test_scores[i] = score(model, X_test, y_test)

        # Saving model for later inspection
        models.append(model['classifier'])

    train_scores = pd.DataFrame(train_scores,columns=score_names)
    test_scores = pd.DataFrame(test_scores,columns=score_names)


    # Prints summary statistics
    results = pd.DataFrame()
    results['train_mean'] = train_scores.mean()
    results['train_stdev'] = train_scores.std()
    results['test_mean'] = test_scores.mean()
    results['test_stdev'] = test_scores.std()
    print("----------------------------------------")
    print('Sample results:')
    with pd.option_context('display.float_format', '{:0.3f}'.format):
        print(results)

    # Plot distribution of hyperparmeters
    for param in param_grid.keys():
        param = param.split('__')[1] # we ignore the 'classifier__' prefix
        ls = [getattr(model, param) for model in models]

        plt.hist(ls)
        plt.title(param)
        plt.savefig(figfolder + param + '_dist.png')
    plt.clf()

    #plotting results (metrics and feature importance)
    if theta == 1:
        visualise_results(train_scores, test_scores, score_names, figfold = figfolder, config_str=" ")
        visualise_feature_importance(models, figfold = figfolder, config_str=" ")
    else:
        visualise_results(train_scores, test_scores, score_names, figfold = figfolder, config_str= f"theta = {theta}")
        visualise_feature_importance(models, figfold = figfolder, config_str= f"theta = {theta}")