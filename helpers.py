import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

##########################
###### CORRELATIONS ######
##########################

######################
###### OUTLIERS ######
######################

# Return min, max values on IQR scores
def outliers_IQR(df, feature):
  try:

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1

    # lower bounds
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return lower, upper
  except Exception as e:
    print("invalid feature")


def outliers_z_score(df, feature, no_z=3):
  lower = df[feature].mean()-no_z*df[feature].std()
  upper = df[feature].mean()+no_z*df[feature].std()
  return lower, upper

# Generalize min max rule used for age
# Returns DF with outliers
def outliers_min_max(df, feature, min=None, max=None):
  try:
    cond_min = df[feature] < min if min != None else False
    cond_max = df[feature] > max if max != None else False
    return df[cond_min | cond_max ]
  except Exception as e:
    print("invalid feature")
    
# Function that can be run on both training set and test set
# to handle outliers.
# Set to Nan if outside boundary
def handle_outliers(df, df_bounds):
  for f in df_bounds.index:
      outliers = outliers_min_max(df, f,
                                  min=df_bounds.loc[f, 'Lower'],
                                  max=df_bounds.loc[f, 'Upper']
                                )
      df.loc[outliers.index, f] = np.NaN
  return df

##########################
###### MISSING DATA ######
##########################

def BMI(weight, height):
  return weight/(height**2/(100*100))


def fix_obesity(df, threshold=30):
  idx = df[df['Obesity'].isna()].index
  
  # indexes to identify BMI above or below threshold
  idx2 = df.loc[idx,].loc[BMI(df.loc[idx,]["Weight"], df.loc[idx,]["Height"]) <= threshold].index
  idx3 = df.loc[idx,].loc[BMI(df.loc[idx,]["Weight"], df.loc[idx,]["Height"]) > threshold].index
  
  # set obesity from indexes above
  df.loc[idx2,'Obesity'] = 0
  df.loc[idx3,'Obesity'] = 1
  return df


def fix_polydipsia(df, threshold=2.5):
  idx = df[df['Polydipsia'].isna()].index
  
  # indexes to identify Urination above or below threshold
  idx2 = df.loc[idx,].loc[df['Urination'] <= threshold].index
  idx3 = df.loc[idx,].loc[df['Urination'] > threshold].index

  # set Polydipsia from indexes above
  df.loc[idx2,'Polydipsia'] = 0
  df.loc[idx3,'Polydipsia'] = 1
  #df.loc[idx,]
  return df

##################
###### MISC ######
##################

def model_summary(clf, X_test, y_test, header = True, name=''):
   # computes the accuracy, precision and recall for a classifier and prints a standard output.
   assert 'predict' in dir(clf), "Classifier must have a 'predict' method"

   y_pred = clf.predict(X_test)

   acc = accuracy_score(y_test, y_pred)
   prec, recall, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
   if header:
      print('Accuracy\tPrecision\tRecall')
   print(f'{acc :.2f}\t\t{prec:.2f}\t\t{recall:.2f}\t{name}')


def combined_outliers(train: pd.DataFrame, features: list, test: pd.DataFrame = None):
   # Calculate combined outliers for continous features using euclidean norm
  assert len(features) > 1
  #train  = train.copy()
  train = train[features]
  train = train.fillna(train.mean())
  train = train.to_numpy()
  train = (train - train.mean())/(train.std()) # normalize to be indepentent of parameterization
  d_train = np.sqrt(np.square(train).sum(axis=1)) #Calculate square distance
  z_train = (d_train-d_train.mean())/d_train.std()   # Normalize distances
  ret = z_train
  # If we get a test set we need to use parameters from the training set
  if test is not None:
     test = test[features].fillna(train.mean())
     test = test.to_numpy()
     test = (test - train.mean())/(train.std()) # normalize to be indepentent of parameterization
     d_test = np.sqrt(np.square(test).sum(axis=1)) #Calculate square distance
     z_test = (d_test-d_test.mean())/d_test.std()   # Normalize distances
     ret = z_test
  return ret

