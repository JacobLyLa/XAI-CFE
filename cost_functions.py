"""
Implement cost functions/objectives here and use them to generate counterfactuals
"""

import numpy as np
import pandas as pd

def create_weighted_wachter2017_cost_function(X: pd.DataFrame, x: pd.DataFrame, y_target: float, model: object, features: list):
    x.reset_index(drop=True, inplace=True)
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
    epsilon = 1e-4
    mad[mad == 0] = epsilon
    x_normalized = (x - medians) / mad
    
    # Apply feature weights to x
    apply_feature_weights(x, features)
    
    # Put categorical features back
    x_normalized[cat_idx] = x[cat_idx]

    # Define Wachter2017 cost function
    def weighted_wachter2017_cost_function(x_prime: pd.DataFrame, lambda_value: float) -> float:
        x_prime.reset_index(drop=True, inplace=True)
        # Normalize x_prime
        x_prime_normalized = (x_prime - medians) / mad
        # Put categorical features back
        x_prime_normalized[cat_idx] = x_prime[cat_idx]
        # apply feature weights to x_prime
        x_prime_normalized = apply_feature_weights(x_prime_normalized, features)
        
        # Compute distances
        numeric_distance = np.sum(np.minimum((x_normalized[num_idx] - x_prime_normalized[num_idx])**2, 100).sum())
        categorical_distance = x_prime_normalized[cat_idx].sum(axis=1).sum()
        
        # Compute misfit
        prediction = model.predict_proba(x_prime)[0][0]
        misfit = (prediction - y_target)**2
        
        return lambda_value * misfit + numeric_distance + categorical_distance * 2
    
    return weighted_wachter2017_cost_function

def create_wachter2017_cost_function(X: pd.DataFrame, x: pd.DataFrame, y_target: float, model: object):
    x.reset_index(drop=True, inplace=True)
    cat_idx = X.select_dtypes(include=['object', 'bool']).columns
    num_idx = X.select_dtypes(include=['int64', 'float64']).columns
    # Compute normalized data
    medians = X.median()
    mad = (X - medians).abs().median()
    epsilon = 1e-4
    mad[mad == 0] = epsilon
    x_normalized = (x - medians) / mad
    
    # Put categorical features back
    x_normalized[cat_idx] = x[cat_idx]

    # Define Wachter2017 cost function
    def wachter2017_cost_function(x_prime: pd.DataFrame, lambda_value: float) -> float:
        x_prime.reset_index(drop=True, inplace=True)
        # Normalize x_prime
        x_prime_normalized = (x_prime - medians) / mad
        
        # Put categorical features back
        x_prime_normalized[cat_idx] = x_prime[cat_idx]
        
        # Compute distances
        numeric_distance = np.sum(np.minimum((x_normalized[num_idx] - x_prime_normalized[num_idx])**2, 10).sum())
        categorical_distance = np.sum(x_normalized[cat_idx] != x_prime_normalized[cat_idx]).sum()
        
        # Compute misfit
        prediction = model.predict_proba(x_prime)[0][0]
        misfit = (prediction - y_target)**2
        
        return lambda_value * misfit + numeric_distance + categorical_distance * 2
    
    return wachter2017_cost_function


if __name__ == "__main__":
    import pickle
    from prefrences import Feature, get_constant_weight_function, get_constant_category_weight_function
    from counterfactuals import get_counterfactuals
    with open('pretrained_models/pipeline_adult.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('datasets/adult.csv', na_values='?')
    df = df.drop(columns = ['fnlwgt'])
    df = df.dropna()
    X = df.drop(columns=['income', 'education'])
    x = X.loc[0:0]
    print("\nOriginal x:")
    print(x)
    print("-"*150)
    y = model.predict_proba(x)[0][0]
    y_target = y - 0.2
    
    features = []
    features.append(Feature('age', x.values[0][0], (30, 50), get_constant_weight_function(5.0)))
    features.append(Feature('workclass',  x.values[0][1], X['workclass'].unique()))
    features.append(Feature('educational-num',  x.values[0][2], (10, 14), get_constant_weight_function(1)))
    features.append(Feature('marital-status',  x.values[0][3], X['marital-status'].unique()))
    features.append(Feature('occupation',  x.values[0][4], X['occupation'].unique()))
    features.append(Feature('relationship',  x.values[0][5], X['relationship'].unique()))
    features.append(Feature('race',  x.values[0][6], X['race'].unique(), get_constant_category_weight_function(10.0, x.values[0][6])))
    features.append(Feature('gender',  x.values[0][7], X['gender'].unique(), get_constant_category_weight_function(1, x.values[0][7])))
    features.append(Feature('capital-gain',  x.values[0][8], (0, 10), get_constant_weight_function(1)))
    features.append(Feature('capital-loss',  x.values[0][9], (0, 10), get_constant_weight_function(1)))
    features.append(Feature('hours-per-week',  x.values[0][10], (0, 70), get_constant_weight_function(1)))
    features.append(Feature('native-country',  x.values[0][11], X['native-country'].unique()))
    
    normal_watcher = create_wachter2017_cost_function(X, x, y_target, model)
    weighted_watcher = create_weighted_wachter2017_cost_function(X, x, y_target, model, features)
    
    # First optimize with normal cost function
    CFS = get_counterfactuals(x, y_target, model, normal_watcher, features, 0.1, 100)
    print("CFS:")
    print(CFS)
    x_prime = CFS.loc[0:0]
    normal_cost = normal_watcher(x_prime, 0.0)
    print("Normal cost", normal_cost)
    weighted_cost = weighted_watcher(x_prime, 0.0)
    print("Weighted cost", weighted_cost)
    
    # Then optimize with weighted cost function
    CFS = get_counterfactuals(x, y_target, model, weighted_watcher, features, 0.05, 100)
    print("CFS:")
    print(CFS)
    x_prime = CFS.loc[0:0]
    print("Checking normal cost for first CF")
    normal_cost = normal_watcher(x_prime, 0.0)
    print("Normal cost", normal_cost)
    print("Checking weighted cost for first CF")
    weighted_cost = weighted_watcher(x_prime, 0.0)
    print("Weighted cost", weighted_cost)

    capital_loss = features[9]
    # hist plot of values
    from matplotlib import pyplot as plt
    values = capital_loss.value_history
    plt.hist(values, bins=20)
    plt.show()

    