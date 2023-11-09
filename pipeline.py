import warnings
warnings.simplefilter(action='ignore', category=UserWarning)    # ignore pesky warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # with version specified, future warnings is superfluous (although this is bad practice)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import Literal

# Pipeline imports
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,FunctionTransformer
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator


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

np.random.seed(2023)


def BMI(weight, height):
    return weight/(height**2/(100*100))

class CustomTransformer(BaseEstimator, TransformerMixin):
    """
    Transofrmer which imputes Obesity and Polydipsia parametrically with BMI and Urination
    """
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        """Expected method by the pipeline API"""
        return self

    def transform(self, X, y=None, threshold_BMI=30, threshold_Polydipsia= 2.5):
        X = self._BMI(X, threshold=threshold_BMI)
        X = self._fix_polydipsia(X, threshold=threshold_Polydipsia)
        return X
    
    def _BMI(self, X, y=None, threshold=30):
        # Perform arbitary transformation
        idx = X[X['Obesity'].isna()].index
        
        # indexes to identify BMI above or below threshold
        idx2 = X.loc[idx,].loc[BMI(X.loc[idx,]["Weight"], X.loc[idx,]["Height"]) <= threshold].index
        idx3 = X.loc[idx,].loc[BMI(X.loc[idx,]["Weight"], X.loc[idx,]["Height"]) > threshold].index
        
        # set obesity from indexes above
        X.loc[idx2,'Obesity'] = 0
        X.loc[idx3,'Obesity'] = 1
        return X
    
    def _fix_polydipsia(self, df, threshold=2.5):
        idx = df[df['Polydipsia'].isna()].index
        
        # indexes to identify Urination above or below threshold
        idx2 = df.loc[idx,].loc[df['Urination'] <= threshold].index
        idx3 = df.loc[idx,].loc[df['Urination'] > threshold].index

        # set Polydipsia from indexes above
        df.loc[idx2,'Polydipsia'] = 0
        df.loc[idx3,'Polydipsia'] = 1
        #df.loc[idx,]
        return df
    
    def set_output(self, *, transform: Literal['default', 'pandas'] | None = None) -> BaseEstimator:
        return super().set_output(transform=transform)


class AddBMI(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        """Expected method by the pipeline API"""
        return self

    def transform(self, X, y=None):
        X['BMI'] = BMI(X['Weight'], X['Height'])
        return X
    
    def set_output(self, *, transform: Literal['default', 'pandas'] | None = None) -> BaseEstimator:
        return super().set_output(transform=transform)

class Outliers(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        """Expected method by the pipeline API"""
        return self

    def transform(self, X: pd.DataFrame, y=None):
        df = X[self.features].copy()
        mask = abs((df - df.mean())/df.std()) > 3 # Standardize
        X[mask] = np.NaN
        return X
    
    def get_feature_names_out(self, *args, **params):
        """
        method which enables pass through of column names, preserving them in the final transformed data frame
        """
        return self.columns_


diabetes = pd.read_csv(infile)
binary_features = ['Obesity', 'TCep', 'Polydipsia', 'Sudden Weight Loss', 'Weakness',
                'Polyphagia', 'Genital Thrush', 'Visual Blurring', 'Itching',
                'Irritability', 'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Alopecia', 'Gender']
cat_features = ['Race',	'Occupation',	'GP']
num_features = ['Age',	'Height',	'Weight',	'Temperature',	'Urination']


target = 'Diabetes'
y = diabetes[target]
X = diabetes.drop(columns=(target))

# If our model has worse accuracy than this, simply guessing 'Postive' is a better model
print(f"Mean diabetes in data set: {(y=='Positive').mean()}") 

# features reomved here will not be included in the analysis
num_features.remove('Urination')
cat_features.remove('GP')
cat_features.remove('Race')
binary_features.remove('TCep')



# # Some helpers
def fix_height(x, threshold=100):
    """ Converts height in meters to centimeters, if height is less than threshold (default = 100)"""
    col = x.columns[0]
    mask = x[col] < threshold
    x.loc[mask, [col]] = x.loc[mask, [col]].mul(100)
    return x

def fix_formating(x):
    return x.replace({'yes':1, 'Yes': 1, 'Positive':1, 'no':0, 'No':0, 'Negative':0, 'Male':1,'Female':0})



# ## $\epsilon$ privacy


# # Constructing Pipeline
# 
# I compose the pipeline from smaller pipelines, which all handles a subset of the tasks.
# 
# The numeric, binary and categorical columns are all handled differently. 
# In addition, construct a parametric preprocessor where we can impute with domain knowledge. We must adapt the functions from the other script to do so, and I have only done that to a few easy ones.
# 
# For transformations which rely on other columns, like fixing obesity and polydipsia, we must use a slightly more complicated approach with classes, which I haven't attempted yet.
# 
# Row wise transformations, like outliers, must also be implemented, and I have not looked at that either.


# Parametric preprocessor where we impute with domain knowledge
preprocessor_parametric = ColumnTransformer(
    transformers=[
        # ('fix height', FunctionTransformer(fix_height), ['Height']),
    ],
    verbose_feature_names_out= False, # Keeps the same column name for future processing
    remainder='passthrough'         # Doesent drop untransformed columns
).set_output(transform='pandas')    # Keep data frame format

binary_transformer = Pipeline(
    steps=[
        ('Fix formating', FunctionTransformer(fix_formating)),
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        # ('randomize', FunctionTransformer(randomize)), # privacy has been moved to a separate pipeline
        # ("selector", SelectKBest(k=5)),
    ]
)

cat_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.1, sparse=False)),
        # Unsure how to introduce privacy,
        # ("selector", SelectKBest(k=5)),
    ]
)

num_transformer = Pipeline(
    steps=[
           # Differential privacy here
           # Outliers Here
            ('Outliers', Outliers(num_features)), 
            ("imputer", SimpleImputer(strategy="mean")), 
            ("scaler", StandardScaler())]
)


# General preprocesser which encodes and scales all features
preprocessor_general = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features),
        ('binary', binary_transformer, binary_features)
    ],
    verbose_feature_names_out= True,
    remainder='drop'                # drop untouched features since after this step, as it is the last preprocessing one
).set_output(transform="pandas")    # Keep data frame format


preprocessor = Pipeline(
    steps=[
        ('Custom impute', CustomTransformer()),
        # ('Add columns', AddBMI()),
        ("preprocessor parametric", preprocessor_parametric), 
        ("preprocessor general", preprocessor_general), 
        ]
)


clf = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ("classifier", tree.DecisionTreeClassifier())
        ]
)


def make_clf(clf):
    return Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ("classifier", clf)
        ]
    )


# Defining the parameter grid which the pipeline optimizes w.r.t.
param_grid = {
    'classifier__ccp_alpha' : np.linspace(0, 0.001, 10),
    'classifier__max_depth' : list(range(1,10))
}

def tune_clf(clf, X_train, y_train, param_grid=param_grid):
    """
    Tunes a model with a grid search of the parameter space, then returns a model trained on the optimal hyperparameters.

    ----
    returns:
    {
        'model',
        'train score'
    }
    """
    model = GridSearchCV(clf, param_grid=param_grid).fit(X_train, y_train)
    best_model = model.best_estimator_.fit(X_train, y_train)
    # print(best_model['classifier'].max_depth)
    return {'model': best_model, 'train score': best_model.score(X_train, y_train)}




# Defining the metrics we want to and join them in a single function
def score(model, X, y):
    """
    Accepts a model (implementing a .predict method) and some data, X, with labels y, 
    and returns the accuracy, precision, recall and F1 score.

    Warning: the lenght of the return value is assumed to be equal to the lenght of the socre names

    """
    y_hat = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    acc = (tp + tn)/(tp + fp + fn + tn)
    prec = (tp)/(tp + fp)
    rec = (tp)/(tp + fn)
    f1 = 2*prec*rec/(prec+rec)


    return acc, prec, rec, f1

# Note: the shape of this list must match the shape of the return value of score().
score_names = ['Accuracy', 'Precision', 'Recall', 'F1']


# Bootstrapping 

# n_samples = 20 # number of bootstrap samples
models = [] # list to save the constructed models
train_scores = np.zeros(shape=(n_samples, len(score_names))) # array to save train scores
test_scores = np.zeros_like(train_scores)   # array to save test scores


"""
Discuss this a lot, since we have thought about it so much. 

Note: we dont correct the bootstrap sample here since it is not clear if the 
0.628 correction is applicable with zero-one loss (and as a consequence accuracy),
and in particular, how we correct other performance measures, like Precission and 
Recall.
"""

# Bootstrap loop
print(f'Generating {n_samples} bootstrap samples')
for i in range(n_samples):
    print(f'{i/n_samples*100:3.0f}% done')

    idx = np.random.choice(range(len(diabetes.index)), size=len(diabetes.index), replace=True)
    assert len(set(idx)) < len(idx) # testing for repeated samples

    train = diabetes.iloc[idx]
    test = diabetes[~diabetes.index.isin(idx)] 

    # testing for complete split
    assert len(test.index) + len(set(idx)) == len(diabetes.index), f"{len(test.index)}, {len(set(idx))}, {len(diabetes.index)}  "
    
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)

    X_train = train.drop(target, axis=1)
    y_train = train[target]
    X_test = test.drop(target, axis=1)
    y_test = test[target]

    model =  tune_clf(make_clf(tree.DecisionTreeClassifier()), X_train, y_train)['model']

    # Making train and test scores
    train_scores[i] = score(model, X_train, y_train)
    test_scores[i] = score(model, X_test, y_test)

    # Saving model for later inspection
    models.append(model['classifier'])

train_scores = pd.DataFrame(train_scores,columns=score_names)
test_scores = pd.DataFrame(test_scores,columns=score_names)


# Prints summary statistics
results = pd.DataFrame()
results['train_means'] = train_scores.mean()
results['train_stdev'] = train_scores.std()
results['test_means'] = test_scores.mean()
results['test_stdev'] = test_scores.std()
print('Sample results')
print(results)

# Plots distribution of hyperparmeters
for param in param_grid.keys():
    param = param.split('__')[1] # we ignore the 'classifier__' prefix
    ls = [getattr(model, param) for model in models]

    plt.hist(ls)
    plt.title(param)
    plt.savefig(figfolder + param + '_dist.png')
plt.clf()

# Plots nice histograms of measures, both on train and test set, in a single figure.
# The figure has two columns, and calculates the neccesary number of rows
fig, axs = plt.subplots(len(score_names)//2 + len(score_names)%2, 2, figsize=(8, 6)) 
for i, score in enumerate(score_names):
    j = i%2 
    k = i//2
    axs[j,k].hist(train_scores[score], label='train')
    axs[j,k].hist(test_scores[score], label='test')
    axs[j,k].set_title(score)
    axs[j,k].legend()
    axs[k,j].set_xlim([0.75,1.01])
    
plt.savefig(figfolder + 'result_hist.png')
plt.clf()


# Visualizing feature importance, using the DecisionTree()'s build in feature importance measure.
# First, all the feature importances must be colleceted, and since the features vary from model
# to model (due to the pipelines automatic selection), it is slightly complicated.

feature_importance = {} # Save them as a dict.
for model in models: # Loop through all models
    fnames = model.feature_names_in_    # get feature names
    fimp = model.feature_importances_   # and corresponding importances

    for i in list(zip(fnames,fimp)):
        if i[0] in feature_importance.keys():
            feature_importance[i[0]] = feature_importance[i[0]] + [i[1]] # add importance to list if exists
        else:
            feature_importance[i[0]] =[i[1]] # create new list if first encounter with feature

# Finally, loop through dict and plot histograms
for feature in feature_importance.keys():
    plt.hist(feature_importance[feature])
    plt.title(feature)
    plt.xlim([-0.01,0.6]) # add common x-axis for ease of comparison.
    plt.savefig(figfolder + feature + '_dist.png')
    plt.clf()
