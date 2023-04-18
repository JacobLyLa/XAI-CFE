"""
Implement cost functions/objectives here and use them to generate counterfactuals
"""
import numpy as np
import scipy.stats

import pandas as pd

def mad_weighted_distance(x_prime, x, X, weight_functions):
    mad = scipy.stats.median_abs_deviation(X, axis=0)
    # weight x and x_prime
    x_prime_weighted = np.array([[weight_functions[i](x_prime.values[0][i]) for i in range(x_prime.shape[1])]])
    x_weighted = np.array([[weight_functions[i](x.values[0][i]) for i in range(x.shape[1])]])
    distance = np.sum(np.abs(x_weighted-x_prime_weighted)/(mad+1))
    return distance

def mad_distance(x_prime, x, X):
    mad = scipy.stats.median_abs_deviation(X, axis=0)
    distance = np.sum(np.abs(x-x_prime)/mad)
    return distance
    
# TODO: gjør alle x prime og x til pandas
def wachter2017_cost_function(x_prime: np.ndarray, x: np.ndarray, y_prime, lambda_value: float, model: object, X, weight_functions) -> float:
    distance = mad_distance(x_prime, x, X)
    prediction = model.predict_proba(x_prime.reshape((1, -1)))[0][0]
    misfit = (prediction-y_prime)**2
    return lambda_value * misfit + distance


def weighted_watcher_cost_function(x_prime: pd.DataFrame, x: pd.DataFrame, y_prime, lambda_value: float, model: object, X, weight_functions) -> float:
    distance = mad_weighted_distance(x_prime.values, x.values, X, weight_functions)
    try:
        #prediction = model.predict_proba(x_prime.reshape((1, -1)))[0][0]
        # x_prime is a dataframe with one row, the model is trained to preict on np.array
        prediction = model.predict_proba(x_prime.values.reshape((1,-1)))[0][0]
    except:
        prediction = model.predict_proba(x_prime)[0][0]
    misfit = (prediction-y_prime)**2
    return lambda_value * misfit + distance


def wachter2017_cost_function_ignore_categorical(x_prime: pd.DataFrame, x: pd.DataFrame, y_prime, lambda_value: float, model: object, X, weighted_functions=None) -> float:
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns  # numerical features
    mad =  scipy.stats.median_abs_deviation(X[num_ix], axis=0)
    if (type(x_prime) == np.ndarray): #the case when we use sklearn optimizer
        distance = np.sum(np.abs(x[num_ix].values-x_prime)/(mad+1))
        x_prime_to_df = x.copy()
        for i in range(len(x_prime)):
            x_prime_to_df.at[x.index[0],num_ix[i]] = x_prime[i] #create dataframe where the values from x_prime is used 
        prediction = model.predict_proba( x_prime_to_df)[0][0]
    else: # when we use optuna x_prime is DataFrame
        distance = np.sum(np.abs(x[num_ix].values-x_prime[num_ix].values)/(mad+1)) #[num_ix].values
        prediction = model.predict_proba(x_prime)[0][0]
    misfit = (prediction-y_prime)**2
    return lambda_value * misfit + distance


def wachter2017_cost_function_with_categorical(x_prime: pd.DataFrame, x: pd.DataFrame, y_prime, lambda_value: float, model: object, X, weighted_functions=None) -> float:
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns  # numerical features
    #mad = scipy.stats.median_abs_deviation(X[num_ix], axis=0)
     # when we use optuna x_prime is DataFrame, TODO: sklearn?
    # numeriske indexer
    num_indexes =  [x_prime.columns.get_loc(col) for col in num_ix] 
    num_weight_functions = np.array(weighted_functions)[[x_prime.columns.get_loc(col) for col in num_ix] ]
    distance_numerical = mad_weighted_distance(x_prime=x_prime[num_ix], x=x[num_ix],X=X[num_ix], weight_functions=num_weight_functions)
    #distance_numerical = np.sum(np.abs(x[num_ix].values-x_prime[num_ix].values)/(mad+1)) #[num_ix].values
    # Calculate categorical distance
    distance_categorical = 0
    for feature in cat_ix:
        if x[feature].values != x_prime[feature].values:
            distance_categorical += 5000 # TODO: its not even weighted...
    prediction = model.predict_proba(x_prime)[0][0]
    misfit = (prediction-y_prime)**2
    return lambda_value * misfit + distance_numerical + distance_categorical