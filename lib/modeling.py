import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR

import scipy.stats as st

def get_cm(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm = cm / cm.astype(float).sum(axis=1)
    np.round_(cm, decimals=4, out=cm)
    
    return cm

def deskew_df(df):
    box_cox_df = pd.DataFrame()
    for col in df.columns:
        box_cox_col = st.boxcox(df[col])[0]
        box_cox_df[col] = pd.Series(box_cox_col)
        
    return box_cox_df

def test_model(my_model, X_train, y_train, X_test, y_test, my_rand_state=42, naive_flag=False, gs_params=None):
    if naive_flag:
        pipe = Pipeline([('scaler', StandardScaler()),
                 ('pca', PCA(random_state=my_rand_state)),
                 ('model', my_model)
                ])
    else:
        pipe = Pipeline([('scaler', StandardScaler()),
                         ('pca', PCA(random_state=my_rand_state)),
                         ('kpca', KernelPCA()),
                         ('model', my_model)
                        ])
    
    if gs_params != None:
        mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        #pipe_gs = GridSearchCV(pipe, param_grid=gs_params, cv=[(slice(None), slice(None))], n_jobs=-1, verbose=1)
        pipe_gs = GridSearchCV(pipe, param_grid=gs_params, cv=[(slice(None), slice(None))], n_jobs=-1, verbose=1, scoring=mae_scorer)
        pipe = pipe_gs
    
    pipe.fit(X_train, y_train)    
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    
    y_pred = pipe.predict(X_test)
    
    print(my_model)
    print('train score:', train_score)
    print('test score:', test_score)
    print()
    
    if gs_params != None:
        return pipe.best_params_

def tts_stock(X, y, test_size=.3):
    if X.shape[0] != y.shape[0]:
        raise ValueError("Error: DataFrames do not contain same row count.")
    
    if type(test_size) != float:
        raise ValueError("Error: Invalid test_size type.")
    
    if test_size <= 0 or test_size >= 1:
        raise ValueError("Error: Invalid test_size value.")
    
    X_sorted = X.sort_index(axis=0)
    y_sorted = y.sort_index(axis=0)
    
    split_index = int(X_sorted.shape[0] * (1 - test_size))
    X_train = X_sorted.iloc[:split_index]
    X_test = X_sorted.iloc[split_index:]
    
    y_train = y_sorted.iloc[:split_index]
    y_test = y_sorted.iloc[split_index:]
    
    return X_train, X_test, y_train, y_test