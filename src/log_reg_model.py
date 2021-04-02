import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import statsmodels.formula.api as smf
from data_pipeline import clean_data

def split_X_y(df, target_col):
    '''
    splits clean df into features, X, and targets, y,
    using specified target column to select targets.

    inputs: clean df, target_col from clean df 
    returns: features (X), targets (y)
    '''
    X = df.copy()
    y = X.pop(target_col)
    return X, y

def confirm_assumptions(X,y):
    '''
    confirms whether sets of independent variables (X)
    and dependent variables (y) meet assumptions of 
    logistic regression.
    inputs: independent variables (X), dependent variables (y)
    returns: prints boolean confirmation of binary dependent
    variable, non-collinearity of independent variables and
    sample-size assumptions of logistic regression.
    '''
    binary_dv = y.nunique() == 2
    corr_mat = X.corr()[X.corr() < 1]
    no_corr = corr_mat.describe()[-1:] >= .4
    empty_series = len(no_corr.any()[no_corr.any() == True])
    low_corr = empty_series == 0
    lrg_samp = (10 * len(X.columns)) / np.mean(y) <= len(y)
    print(f'Binary dependent variable:{binary_dv} \
    Correlation below 0.4: {low_corr} \
    Large sample size: {lrg_samp}')

def find_coefs(X, y, i):
    '''
    evaluates logistic regression model on various feature 
    configurations for coefficient values and scoring metrics
    inputs: features (X), targets (y), index value for formula
    used for regression (i =
                            0: trivial features of voter status and age
                            1: racial demographic features
                            2: gender demographic features
                            3: county composition features
                            4: congressional district geographic features
                            5: complete feature set)
    returns: summary statistics of model
    '''
    
    groupings = ['y~ C(voter_status) + age',
                'y~ C(AI) + C(AP) + C(BH) + C(HP) + C(OT) + C(U) + C(WH)',
                'y~ C(F) + C(M) + C(O)',
                'y~ C(rural) + C(urban) + C(military)',
                'y~ C(cd_1) + C(cd_2) + C(cd_3) + C(cd_4) + C(cd_5) + C(cd_6) + C(cd_7) + C(cd_8) + C(cd_9) + C(cd_10) + C(cd_11) + C(cd_12) + C(cd_13) + C(cd_14) + C(cd_99999)',
                'y~ C(voter_status) + C(AI) + C(AP) + C(BH) + C(HP) + C(OT) + C(U) + C(WH) + C(F) + C(M) + C(O) + C(rural) + C(urban) + C(military) + C(cd_1) + C(cd_2) + C(cd_3) + C(cd_4) + C(cd_5) + C(cd_6) + C(cd_7) + C(cd_8) + C(cd_9) + C(cd_10) + C(cd_11) + C(cd_12) + C(cd_13) + C(cd_14) + C(cd_99999) + age']
    model = smf.logit(formula=groupings[i], data= X).fit()
    return model.summary()

def lr_model(X, y):
    '''
    preliminary setup for sk-learn logistic regression model
    inputs: features (X), targets (y)
    returns: train test splits for features and targets
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    regressor = LogisticRegression(class_weight = 'balanced',
                                   solver='liblinear')
    regressor.fit(X_train, y_train)
    yhat_probs = regressor.predict_proba(X_test)[:,1]

    return X_train, X_test, y_train, y_test


if __name__=='__main__':
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

    clean = clean_data(train)

    X, y = split_X_y(clean, 'new_registration')

    confirm_assumptions(X,y)

    X_train, X_test, y_train, y_test = lr_model(X,y)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    summary = find_coefs(X,y, 3)
    print(summary)