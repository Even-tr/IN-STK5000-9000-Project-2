import warnings
warnings.simplefilter('always', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys 

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import tree


# Local imports
from helpers import outliers_IQR, outliers_z_score,  handle_outliers, fix_obesity
from helpers import combined_outliers
from helpers import BMI, fix_polydipsia 

# Reproducibility
np.random.seed(2023)


# #################################################
# ######## 2. DATA ANALYSIS AND PROCESSING ########
# #################################################

try:
    infile = sys.argv[1] # reac command line argument
except IndexError:
    # default file is diabetes.csv
    infile = 'diabetes.csv'


try:
    diabetes = pd.read_csv(infile)

# Handle imporper filenames
except FileNotFoundError: 
    print(f'\n\nERROR: file {infile} was not found')
    quit()


binary_features = ['Obesity', 'TCep', 'Polydipsia', 'Sudden Weight Loss', 'Weakness',
                   'Polyphagia', 'Genital Thrush', 'Visual Blurring', 'Itching',
                   'Irritability', 'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Alopecia']
cat_features = ['Race',	'Occupation',	'GP']
num_features = ['Age',	'Height',	'Weight',	'Temperature',	'Urination']
one_hot_features = {} # for future use


# #######################################
# ######## INITIAL DATA ANALYSIS ########
# #######################################

"""
# Code used for answering questions in this section
"""

# ### Data Cleaning
# #### Uniform formatting

# Converts all binary features to lower case
for f in binary_features:
   diabetes[f] = diabetes[f].str.lower()

# Converts all binary features to ints, preserving Na-s
diabetes = diabetes.replace({'yes':1, 'no':0}) 
diabetes = diabetes.replace({'Positive':1, 'Negative':0})

# #### Duplicates

# We delete them. We assume they are caused by an error in the data collection, and it's unlikely that there are two correct instances with the exact same values.  
diabetes = diabetes.drop_duplicates(keep='first')

# #### Meters to centimeters
condition = diabetes['Height'] < 100
diabetes.loc[condition, ['Height']] = diabetes.loc[condition, ['Height']].mul(100)

# #### Dropping females and Non-Whites
# Note missing values will also be dropped
diabetes = diabetes[diabetes['Gender'] == 'Male']
diabetes = diabetes[diabetes['Race'] == 'White']


# ### Missing categorical data
# If we don't fill missing categorical data now, we run the risk that either the train or the test set don't contain any NA-s. This can cause a difference in columns after one hot encoding and lead to a crash.
# we want to fill early s.t. there are no Na-s beyond this point
diabetes[cat_features + ['Gender']] = diabetes[cat_features+ ['Gender']].fillna('MISSING')

# ## Train - Test split
# We split as early as possible to avoid cross contamination of information from the test set.
# ### Splitting

train_proportion = 0.8
train_idx = np.random.choice(diabetes.index, int(train_proportion*len(diabetes.index)), replace=False)
train = diabetes.loc[train_idx]
test = diabetes.drop(train_idx)
assert len(diabetes.index) == len(train.index) + len(test.index)

# ##########################
# ######## OUTLIERS ########
# ##########################


# #### Identify Boundaries

# Init with domain knowledge
train_outlier_bounds = pd.DataFrame(
    {'Lower': [16, 110, 30, np.nan, np.nan],
     'Upper': [120, 240, 200, np.nan, np.nan]},
     index=num_features
)

# Statistical
for f in ['Temperature', 'Urination']:
    l_IQR, u_IQR = outliers_IQR(train, f)
    l_Z, u_Z = outliers_z_score(train, f)
    train_outlier_bounds.loc[f] = [max(min(l_IQR, l_Z),0), max(u_IQR, u_Z)]


### Deal with

train = handle_outliers(train, train_outlier_bounds)
test = handle_outliers(test, train_outlier_bounds)

# How does it look now? All min max values sensible ..
# Box plots after handling

# ### Combined outliers
# Combined outliers must be handled after fixing the individual ones, otherwise the same ones would be discovered

zs_train = combined_outliers(train[num_features], num_features)



# #### Handle
train = train[zs_train < 4]
zs_test = combined_outliers(test, num_features, test.copy())
test = test[zs_test < 4]

# #####################################
# ######## MISSING DATA PART 2 ########
# #####################################

# ### Deal with

# #### Derived Features

train = fix_obesity(train)
test = fix_obesity(test)

train = fix_polydipsia(train)
test = fix_polydipsia(test)

# #### Missing Binaries

train[binary_features] = train[binary_features].fillna(0)
test[binary_features] = test[binary_features].fillna(0)

# #### Missing numeric

# Fill Na-s with mean. 
train[num_features] = train[num_features].fillna(train[num_features].mean())
# We fill the test data with the mean of the train data, making the test set indepentent of each others
test[num_features] = test[num_features].fillna(train[num_features].mean())

# #### Testing for remaining Na-s

assert train.isna().sum().sum() == 0, f'train data still containts {train.isna().sum().sum()} Na-s'
assert test.isna().sum().sum() == 0, f'test data still containts {test.isna().sum().sum()} Na-s'

# Henceforth, we may assume both train and test data contains no Na-s, drastically simplifying the rest of the code. For online learning, we should perhaps implement the code in this section as a function so that we can apply it on new cases continuously.

# ## Encoding

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



# ###################################
# ######## DECISION TREE ############
# ###################################

clf = tree.DecisionTreeClassifier(max_depth=7)
clf_full_tree = clf.fit(X_train, y_train)

y_test_pred = clf_full_tree.predict(X_test)

confusion_mat = metrics.confusion_matrix(y_test, y_test_pred)
con_mat_disp = ConfusionMatrixDisplay(confusion_mat, display_labels=clf.classes_)


# ### Some Pruning
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
#print(f'Train score {accuracy_score(y_train_pred,y_train)}')
print(f'Test Accuarcy {accuracy_score(y_test_pred,y_test)}')

tree.plot_tree(clf_pruned_tree, feature_names=selected_features, class_names=classes, filled=True)