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

def test_metric(X, x_list, y_list, y_prime_list, features_list, metric, model, steps):
    results = []
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]
        y_target = y_prime_list[i]
        features = features_list[i]
        # the cost function to optimize CFS on  
        #cost_func = create_wachter2017_cost_function(X, x, y_target, model)
        cost_func = create_weighted_wachter2017_cost_function(X, x, y_target, model, features)
        
        print(f"Testing {i+1}/{len(x_list)}")
        # get the counterfactual
        CFS = get_counterfactuals(x, y_target, model, cost_func, features, 0.1, steps) # TODO: tolerance dynamic
        
        # get best counterfactual if there are any
        # TODO: maybe a metric that checks how many time it fails generating any counterfactuals
        if len(CFS) == 0:
            continue

        # best CF. TODO: do multiple CFS
        x_prime = CFS.loc[0:0]
        results.append(metric(X, x, x_prime, y, y_target, model, features))
    return np.array(results)

def closeness(X, x, x_prime, y, y_target, features):
    # calculate absolute difference between y and y_target
    return abs(y - y_target)

'''
Takes the original x and x_prime and returns the amount of features that have been changed
'''
def sparsity(X, x, x_prime, y, y_target, model, features):
    # count the number of features that have been changed
    count = 0
    x_values = x.values[0]
    x_prime_values = x_prime.values[0]
    for i in range(len(x_values)):
        if x_values[i] != x_prime_values[i]:
            count += 1
    return count / len(x_values)

'''
Takes the original x and x_prime and returns the closest datapoint in X
'''
def distance_closest(X, x, x_prime, y, y_target, model, features):
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

def generate_individual(X, model, visualize=False):
    x_index = random.randint(0, len(X)-1)
    x = X.loc[x_index:x_index]
    # calculate y, from the model
    y = model.predict_proba(x)[0][0]
    # adjust y to +- 0.2 probabilisticly, and make sure it is between 0 and 1
    y_target = y + random.uniform(-0.2, 0.2)
    y_target = max(0.0, min(1.0, y_target))

    # create personalized features
    features = []

    #TODO: make it 50% chance of weight feature is not 1

    feature_name = 'age' # can only increase
    lower_boundry = int(x[feature_name])
    upper_boundry = X[feature_name].max()
    value = x.values[0][0]
    features.append(Feature(feature_name, value, int, (lower_boundry, upper_boundry), get_constant_weight_function(random.uniform(0.1, 5))))

    feature_name = 'workclass'
    value = x.values[0][1]
    features.append(Feature(feature_name, value, object, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 5), value)))

    feature_name = 'educational-num' # can only increase
    lower_boundry = int(x[feature_name])
    upper_boundry = X[feature_name].max()
    value = x.values[0][2]
    features.append(Feature(feature_name, value, int, (lower_boundry, upper_boundry), get_constant_weight_function(random.uniform(0.1, 5))))

    feature_name = 'marital-status'
    value = x.values[0][3]
    features.append(Feature(feature_name, value, object, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 5), value)))
    
    feature_name = 'occupation'
    value = x.values[0][4]
    features.append(Feature(feature_name, value, object, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 5), value)))

    feature_name = 'relationship'
    value = x.values[0][5]
    features.append(Feature(feature_name, value, object, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 5), value)))

    feature_name = 'race'
    value = x.values[0][6]
    features.append(Feature(feature_name, value, object, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 5), value)))
    
    feature_name = 'gender'
    value = x.values[0][7]
    features.append(Feature(feature_name, value, object, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 5), value)))

    ### TODO: these contribute so much to distance since median is 0
    feature_name = 'capital-gain'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    value = x.values[0][8]
    features.append(Feature(feature_name, value, int, (lower_boundry, 1), get_constant_weight_function(random.uniform(0.1, 5)))) # change to 0 for now probably

    feature_name = 'capital-loss'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    value = x.values[0][9]
    features.append(Feature(feature_name, value, int, (lower_boundry, 1), get_constant_weight_function(random.uniform(0.1, 5))))

    feature_name = 'hours-per-week'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    value = x.values[0][10]
    features.append(Feature(feature_name, value, int, (lower_boundry, upper_boundry), get_constant_weight_function(random.uniform(0.1, 5))))

    feature_name = 'native-country'
    value = x.values[0][11]
    features.append(Feature(feature_name, value, object, X[feature_name].unique(), get_constant_category_weight_function(random.uniform(0.1, 5), value)))
    
    # visualize
    if visualize:
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
    # fix indices
    df = df.reset_index(drop=True)
    X = df.drop(columns=['income', 'education'])
    y = df['income']

    metrics_to_test = [CF_distance, CF_distance_weighted, sparsity]
    # create n random examples
    x_list = []
    y_list = []
    y_prime_list = []
    features_list = []
    individuals = 10
    for i in range(individuals):
        x, y, y_target, features = generate_individual(X, model, visualize=False)
        x_list.append(x)
        y_list.append(y)
        y_prime_list.append(y_target)
        features_list.append(features)
        
    # for each metric evaluate all examples
    for metric in metrics_to_test:
        results = test_metric(X, x_list, y_list, y_prime_list, features_list, metric, model, 200)
        fail_count = individuals - len(results)
        print("Results for metric:", metric.__name__)
        print(results)
        print("Average:", results.mean())
        print("Standard deviation:", results.std())
        print("Median:", np.median(results))
        print("Failed individuals:", fail_count)
        print("-"*40)

if __name__ == "__main__":
    test()