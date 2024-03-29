import pandas as pd
import numpy as np
import sys
from typing import Literal

# Pipeline imports
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler,FunctionTransformer
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

# Global constants
binary_features = ['Obesity', 'TCep', 'Polydipsia', 'Sudden Weight Loss', 'Weakness',
                'Polyphagia', 'Genital Thrush', 'Visual Blurring', 'Itching',
                'Irritability', 'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Alopecia', 'Gender']
cat_features = ['Race',	'Occupation',	'GP']
num_features = ['Age',	'Height', 'Weight', 'Temperature', 'Urination']

# Feature selection
num_features.remove('Urination')
num_features.remove('Temperature')
cat_features.remove('GP')
cat_features.remove('Occupation')
binary_features.remove('TCep')

target = 'Diabetes'

def BMI(weight, height):
    """
    Calculates BMI from weight in kg and height in cm
    """

    return weight/(height**2/(100*100))

class CustomTransformer(BaseEstimator, TransformerMixin):

    """
    Custom Transformer class which imputes Obesity and Polydipsia parametrically with BMI and Urination

 
    Attributes:
        BaseEstimator: Base class for all estimators in scikit-learn
        TransformerMixin: Mixin class for all transformers in scikit-learn.
     ----------------
     Methods:
        fit(self, X, y=None): Fit transformer by checking X. 
        _BMI(self, X, y=None, threshold=30): Imputes Obesity from BMI for missing obseity values
        _fix_polydipsia(self, df, threshold=2.5): Imputes Polydipsia from Urination for missing polydipsia values
        transform(self, X, y=None, threshold_BMI=30, threshold_Polydipsia= 2.5): Using the forward function to set X 
        set_output(self, *, transform) -> BaseEstimator: Set output to pandas dataframe

    """

   
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        """Expected method by the pipeline API"""
        return self
    
    def _BMI(self, X, y=None, threshold=30):
        # It first identifies missing values in the 'Obesity' column.
        idx = X[X['Obesity'].isna()].index
        
        #For missing values, calculate BMI from the 'Weight' and 'Height' columns
        # and identify BMI above or below threshold
        idx2 = X.loc[idx,].loc[BMI(X.loc[idx,]["Weight"], X.loc[idx,]["Height"]) <= threshold].index
        idx3 = X.loc[idx,].loc[BMI(X.loc[idx,]["Weight"], X.loc[idx,]["Height"]) > threshold].index
        
        # set obesity from indexes above
        X.loc[idx2,'Obesity'] = 0
        X.loc[idx3,'Obesity'] = 1
        return X
    
    def _fix_polydipsia(self, df, threshold=2.5):

        #Identify missing values for 'Polydipsia' column
        idx = df[df['Polydipsia'].isna()].index
        
        # indexes to identify Urination above or below threshold
        idx2 = df.loc[idx,].loc[df['Urination'] <= threshold].index
        idx3 = df.loc[idx,].loc[df['Urination'] > threshold].index

        # set Polydipsia from indexes above
        df.loc[idx2,'Polydipsia'] = 0
        df.loc[idx3,'Polydipsia'] = 1
        #df.loc[idx,]
        return df
    
    def transform(self, X, y=None, threshold_BMI=30, threshold_Polydipsia= 2.5):
        X = self._BMI(X, threshold=threshold_BMI)
        X = self._fix_polydipsia(X, threshold=threshold_Polydipsia)
        return X
   
    
    def set_output(self, *, transform) -> BaseEstimator:
        return super().set_output(transform=transform)

class UniformFormatting(BaseEstimator, TransformerMixin):

    """
    UniformFormatting class which converts meters to centimeters and binary answers to 1 and 0.
 
    Attributes:
        BaseEstimator: Base class for all estimators in scikit-learn
        TransformerMixin: Mixin class for all transformers in scikit-learn.
     ----------------
     Methods:
        fit(self, X, y=None): Fit transformer by checking X. 
        _fix_height(self, x, threshold=100): Converts height given in centimeters to meters 
        _fix_formating(self, x): Converts binary answers to 1 and 0
        transform(self, X, y=None, threshold_Height=100): Using the forward function to set X 
        set_output(self, *, transform: Literal['default', 'pandas'] | None = None) -> BaseEstimator: Set output to pandas dataframe

    """


    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        """Expected method by the pipeline API"""
        return self

    # preprocessing functions
    def _fix_height(self, x, threshold=100):

        #col = x.columns[0]
        mask = x['Height'] < threshold
        x.loc[mask, ['Height']] = x.loc[mask, ['Height']].mul(100)
        return x

    def _fix_formating(self, x):
    
        return x.replace({'yes':1, 'Yes': 1, 'Positive':1, 'no':0, 'No':0, 'Negative':0, 'Male':1,'Female':0})

    def transform(self, X, y=None, threshold_Height=100):
        X = self._fix_height(X, threshold=threshold_Height)
        X = self._fix_formating(X)
        return X
   
    
    def set_output(self, *, transform: Literal['default', 'pandas'] | None = None) -> BaseEstimator:
        return super().set_output(transform=transform)
        

class Outliers(BaseEstimator, TransformerMixin):
    """
    Custom Transformer class which removes outliers from the data frame

    Attributes:
        BaseEstimator: Base class for all estimators in scikit-learn
        TransformerMixin: Mixin class for all transformers in scikit-learn.
        ----------------
    Methods:
        fit(self, X, y=None): Fit transformer by checking X. 
        transform(self, X, y=None): Replace outliers with NaN
        get_feature_names_out(self, *args, **params): method which enables pass through of column names, preserving them in the final transformed data frame
    """


    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        """Expected method by the pipeline API"""
        return self

    def transform(self, X: pd.DataFrame, y=None):

        """
        Replace outliers with missing values.
        Rows where the absolute z-score is greater than 3 are marked as outliers.

        Parameters: 
            X (pd.DataFrame): data frame to be transformed

        Returns:
            X (pd.DataFrame): transformed data frame
        """
        df = X[self.features].copy() # copy to avoid changing original data frame

        # Outlier detection
        #  z-score normalization to create a mask of outliers
        mask = abs((df - df.mean())/df.std()) > 3 # Standardize
        # replaces identified outliers in the original DataFrame X with missing values.
        X[mask] = np.NaN
        return X
    
    def get_feature_names_out(self, *args, **params):
        """
        method which enables pass through of column names, preserving them in the final transformed data frame
        """
        return self.columns_

binary_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
    ]
)

cat_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.1, sparse=False)),
    ]
)

num_transformer = Pipeline(
    steps=[
        ('Outliers', Outliers(num_features)), 
        # We are aware that we should impute with median, but we kept mean imputing to allow for a better comparison with project 1.
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
        ('Uniform formatting', UniformFormatting()),
        ('Custom impute', CustomTransformer()), 
        ("preprocessor general", preprocessor_general), 
        ]
)


clf = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ("classifier", tree.DecisionTreeClassifier())
        ]
)

# make model
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
