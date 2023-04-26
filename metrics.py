import pickle
import random
import warnings

import numpy as np
import pandas as pd

from cost_functions import (create_weighted_wachter2017_cost_function, create_wachter2017_cost_function)
from counterfactuals import get_counterfactuals
from prefrences import (Feature, get_constant_weight_function,
                        get_pow_weight_function, get_constant_category_weight_function)

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.5f}'.format
np.random.seed(42)
random.seed(42)

'''
'''
def misfit(X, x, x_prime, y, y_target, model, features):
    # calculate absolute difference between y_target and y_prime_prediction
    y_prime_prediction = model.predict_proba(x_prime)[0][0]
    return abs(y_target - y_prime_prediction)

'''
returns the amount of features that have not been changed
'''
def sparsity(X, x, x_prime, y, y_target, model, features):
    same_count = 0
    x_values = x.values[0]
    x_prime_values = x_prime.values[0]
    for i in range(len(x_values)):
        if x_values[i] == x_prime_values[i]:
            same_count += 1
    return same_count / len(x_values)

'''
returns the closest datapoint from x_prime to any point in X
'''
def distance_closest(X, x, x_prime, y, y_target, model, features):
    raise NotImplementedError
    candidates = X.copy()
    # remove str cols in X and x_prime
    for i in range(len(x)):
        if features[i].variable_type == object:
            candidates = candidates.drop(features[i].name, axis=1)
            x_prime = x_prime.drop(features[i].name, axis=0)
    # calculate the distance
    diff = (candidates - x_prime).abs()
    print(diff.sum(axis=1).min())
    return diff.sum(axis=1).min()

def CF_distance(X, x, x_prime, y, y_target, model, features):
    watcher2017 = create_wachter2017_cost_function(X, x, y_target, model)
    return watcher2017(x_prime, 0) # pass lambda=0 to get distance
    
def CF_distance_weighted(X, x, x_prime, y, y_target, model, features):
    watcher2017_weighted = create_weighted_wachter2017_cost_function(X, x, y_target, model, features)
    return watcher2017_weighted(x_prime, 0) # pass lambda=0 to get distance

def generate_individual(X, model, verbose=False):
    x_index = random.randint(0, len(X)-1)
    x = X.loc[x_index:x_index]
    # calculate y, from the model
    y = model.predict_proba(x)[0][0]
    # adjust y to +- 0.2 probabilisticly, and make sure it is between 0 and 1
    y_target = y + random.uniform(-0.2, 0.2)
    # if it is over 1, subtract that value instead
    if y_target > 1:
        y_target -= 2*(y_target-y)
    # if it is under 0, add that value instead
    elif y_target < 0:
        y_target += 2*(y-y_target)

    # create personalized features
    features = []

    #TODO: make it 50% chance of weight feature is not 1

    feature_name = 'age' # can only increase
    lower_boundry = int(x[feature_name])
    upper_boundry = X[feature_name].max()
    value = x.values[0][0]
    features.append(Feature(feature_name, value, (lower_boundry, upper_boundry), get_constant_weight_function(random.uniform(0.1, 10))))

    feature_name = 'workclass'
    value = x.values[0][1]
    features.append(Feature(feature_name, value, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 10), value)))

    feature_name = 'educational-num' # can only increase
    lower_boundry = int(x[feature_name])
    upper_boundry = X[feature_name].max()
    value = x.values[0][2]
    features.append(Feature(feature_name, value, (lower_boundry, upper_boundry), get_constant_weight_function(random.uniform(0.1, 10))))

    feature_name = 'marital-status'
    value = x.values[0][3]
    features.append(Feature(feature_name, value, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 10), value)))
    
    feature_name = 'occupation'
    value = x.values[0][4]
    features.append(Feature(feature_name, value, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 10), value)))

    feature_name = 'relationship'
    value = x.values[0][5]
    features.append(Feature(feature_name, value, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 10), value)))

    feature_name = 'race'
    value = x.values[0][6]
    features.append(Feature(feature_name, value, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 10), value)))
    
    feature_name = 'gender'
    value = x.values[0][7]
    features.append(Feature(feature_name, value, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 10), value)))

    # TODO: these contribute so much to distance since median is 0. currently set to 0 for no change
    feature_name = 'capital-gain'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    value = x.values[0][8]
    features.append(Feature(feature_name, value, (lower_boundry, 0), get_constant_weight_function(random.uniform(0.1, 10))))

    feature_name = 'capital-loss'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    value = x.values[0][9]
    features.append(Feature(feature_name, value, (lower_boundry, 0), get_constant_weight_function(random.uniform(0.1, 10))))

    feature_name = 'hours-per-week'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    value = x.values[0][10]
    features.append(Feature(feature_name, value, (lower_boundry, upper_boundry), get_constant_weight_function(random.uniform(0.1, 10))))

    feature_name = 'native-country'
    value = x.values[0][11]
    features.append(Feature(feature_name, value, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 10), value)))
    
    # verbose
    if verbose:
        print('x:')
        print(x)
        print()
        print('y:', y)
        print('y_target:', y_target)
        print('features:')
        print("-"*50)
        for feature in features:
            print(feature)
        print("-"*50)

    return x, y, y_target, features

def test():
    with open('pretrained_models/pipeline_adult.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('datasets/adult.csv', na_values='?')
    df = df.drop(columns = ['fnlwgt'])
    df = df.dropna()
    df = df.reset_index(drop=True)
    X = df.drop(columns=['income', 'education'])
    y = df['income']

    metrics_to_test = [misfit, CF_distance, CF_distance_weighted, sparsity]
    results = [[] for _ in range(len(metrics_to_test))]

    # create n random individuals
    individuals = 10
    print(f"Testing {individuals} individuals")
    for i in range(individuals):
        print(i+1, end=" ")
        x, y, y_target, features = generate_individual(X, model, verbose=False)

        # choose cost function to optimize CFS on:

        #cost_func = create_wachter2017_cost_function(X, x, y_target, model)
        cost_func = create_weighted_wachter2017_cost_function(X, x, y_target, model, features)

        CFS = get_counterfactuals(x, y_target, model, cost_func, features, tol=0.1, optimization_steps=500) # can change tol and steps
        
        # get best counterfactual if there are any
        if len(CFS) == 0:
            print("No counterfactuals found for this individual, try increase tol or steps")
            continue

        # test all CFS on each metric
        for j in range(len(CFS)):
            for k, metric in enumerate(metrics_to_test):
                results[k].append(metric(X, x, CFS.loc[j:j], y, y_target, model, features))
    print()
        
    # for each metric evaluate all examples
    for i in range(len(metrics_to_test)):
        result = np.array(results[i])
        fail_count = individuals - len(results)
        print("Results for metric:", metrics_to_test[i].__name__)
        print(result)
        print("Average:", result.mean())
        print("Standard deviation:", result.std())
        print("Median:", np.median(result))
        print("Failed individuals:", fail_count)
        print("-"*40)
        # TODO: save result for notebook analysis

if __name__ == "__main__":
    test()