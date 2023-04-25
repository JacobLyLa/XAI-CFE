"""
Implement cost functions/objectives here and use them to generate counterfactuals
"""

import numpy as np
import pandas as pd
import scipy.stats
import pickle
from prefrences import Feature, get_constant_weight_function, get_constant_category_weight_function


def create_weighted_wachter2017_cost_function(X: pd.DataFrame, x: pd.DataFrame, y_prime: float, model: object, features: list):
    def apply_feature_weights(X, features):
        X_weighted = X.copy()
        for feature in features:
            X_weighted[feature.name] = X[feature.name].apply(feature.weight_function)
        return X_weighted
    
    cat_idx = X.select_dtypes(include=['object', 'bool']).columns
    num_idx = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Compute normalized data
    medians = X.median()
    mad = (X - medians).abs().median()
    epsilon = 1e-6
    mad[mad == 0] = epsilon
    x_normalized = (x - medians) / mad
    
    # Apply feature weights to x
    apply_feature_weights(x, features)
    
    # Put categorical features back
    x_normalized[cat_idx] = x[cat_idx]

    # Define Wachter2017 cost function
    def weighted_wachter2017_cost_function(x_prime: pd.DataFrame, lambda_value: float) -> float:
        # Normalize x_prime
        x_prime_normalized = (x_prime - medians) / mad
        # Put categorical features back
        x_prime_normalized[cat_idx] = x_prime[cat_idx]
        # apply feature weights to x_prime
        x_prime_normalized = apply_feature_weights(x_prime_normalized, features)
        
        # Compute numeric and categorical distances
        numeric_distance = np.sum((x_normalized[num_idx] - x_prime_normalized[num_idx])**2).sum()
        categorical_distance = x_prime_normalized[cat_idx].sum(axis=1).sum()
        
        if lambda_value == 1.1:
            print("numeric_distance", numeric_distance)
            print("categorical_distance", categorical_distance)
        
        # Compute misfit
        prediction = model.predict_proba(x_prime)[0][0]
        misfit = (prediction - y_prime)**2
        
        return lambda_value * misfit + numeric_distance + categorical_distance * 2
    
    return weighted_wachter2017_cost_function

def create_wachter2017_cost_function(X: pd.DataFrame, x: pd.DataFrame, y_prime: float, model: object):
    cat_idx = X.select_dtypes(include=['object', 'bool']).columns
    num_idx = X.select_dtypes(include=['int64', 'float64']).columns
    # Compute normalized data
    medians = X.median()
    mad = (X - medians).abs().median()
    epsilon = 1e-6
    mad[mad == 0] = epsilon
    x_normalized = (x - medians) / mad
    
    # Put categorical features back
    x_normalized[cat_idx] = x[cat_idx]

    # Define Wachter2017 cost function
    def wachter2017_cost_function(x_prime: pd.DataFrame, lambda_value: float) -> float:
        # Normalize x_prime
        x_prime_normalized = (x_prime - medians) / mad
        
        # Put categorical features back
        x_prime_normalized[cat_idx] = x_prime[cat_idx]
        
        # Compute numeric and categorical distances
        numeric_distance = np.sum((x_normalized[num_idx] - x_prime_normalized[num_idx])**2).sum()
        categorical_distance = np.sum(x_normalized[cat_idx] != x_prime_normalized[cat_idx]).sum()
        
        if lambda_value == 1.1:
            print("numeric_distance", numeric_distance)
            print("categorical_distance", categorical_distance)
        
        # Compute misfit
        prediction = model.predict_proba(x_prime)[0][0]
        misfit = (prediction - y_prime)**2
        
        return lambda_value * misfit + numeric_distance + categorical_distance * 2
    
    return wachter2017_cost_function


if __name__ == "__main__":
    from counterfactuals import get_counterfactuals
    with open('pretrained_models/pipeline_adult.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('datasets/adult.csv', na_values='?')
    df = df.drop(columns = ['fnlwgt'])
    df = df.dropna()
    X = df.drop(columns=['income', 'education'])
    x = X.loc[0:0]
    print("Original:")
    print(x)
    y = model.predict_proba(x)[0][0]
    x_prime = x.copy()
    x_prime['age'] = 40
    x_prime['race'] = 'White'
    x_prime['gender'] = 'Female'
    y_prime = y - 0.1
    
    features = []
    features.append(Feature('age', x.values[0][0], float, (30, 50), get_constant_weight_function(5.0)))
    features.append(Feature('workclass',  x.values[0][1], object, X['workclass'].unique()))
    features.append(Feature('educational-num',  x.values[0][2], int, (10, 14), get_constant_weight_function(1)))
    features.append(Feature('marital-status',  x.values[0][3], object, X['marital-status'].unique()))
    features.append(Feature('occupation',  x.values[0][4], object, X['occupation'].unique()))
    features.append(Feature('relationship',  x.values[0][5], object, X['relationship'].unique()))
    features.append(Feature('race',  x.values[0][6], object, X['race'].unique(), get_constant_category_weight_function(10.0, x.values[0][6])))
    features.append(Feature('gender',  x.values[0][7], object, X['gender'].unique(), get_constant_category_weight_function(1, x.values[0][7])))
    features.append(Feature('capital-gain',  x.values[0][8], int, (0, 10), get_constant_weight_function(1)))
    features.append(Feature('capital-loss',  x.values[0][9], int, (0, 10), get_constant_weight_function(1)))
    features.append(Feature('hours-per-week',  x.values[0][10], int, (0, 70), get_constant_weight_function(1)))
    features.append(Feature('native-country',  x.values[0][11], object, X['native-country'].unique()))
    
    normal_watcher = create_wachter2017_cost_function(X, x, y_prime, model)
    weighted_watcher = create_weighted_wachter2017_cost_function(X, x, y_prime, model, features)
    
    # First optimize with normal cost function
    CFS = get_counterfactuals(x, y_prime, model, normal_watcher, features, 0.1, 500)
    print("CFS:")
    print(CFS)
    x_prime = CFS.loc[0:0]
    print("Checking normal cost for first CF")
    normal_cost = normal_watcher(x_prime, 1.0)
    print("Normal cost", normal_cost)
    print("Checking weighted cost for first CF")
    weighted_cost = weighted_watcher(x_prime, 1.0)
    print("Weighted cost", weighted_cost)
    
    # Then optimize with weighted cost function
    CFS = get_counterfactuals(x, y_prime, model, weighted_watcher, features, 0.1, 500)
    print("CFS:")
    print(CFS)
    x_prime = CFS.loc[0:0]
    print("Checking normal cost for first CF")
    normal_cost = normal_watcher(x_prime, 1.1)
    print("Normal cost", normal_cost)
    print("Checking weighted cost for first CF")
    weighted_cost = weighted_watcher(x_prime, 1.1)
    print("Weighted cost", weighted_cost)
    