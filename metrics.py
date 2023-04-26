import os
import pickle
import random
import warnings
import copy
import numpy as np
import pandas as pd

from cost_functions import (create_wachter2017_cost_function,
                            create_weighted_wachter2017_cost_function)
from counterfactuals import get_counterfactuals
from prefrences import (Feature, get_constant_category_weight_function,
                        get_constant_weight_function, get_pow_weight_function)

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.4f}'.format
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})
np.random.seed(42)
random.seed(42)

'''
'''
def within_bounds(X, x, x_prime, y, y_target, model, features):
    # checks how many features that are within the bounds of the original data
    raise NotImplementedError

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
    return watcher2017(x_prime, 0) # lambda=0 to get distance
    
def CF_distance_weighted(X, x, x_prime, y, y_target, model, features):
    watcher2017_weighted = create_weighted_wachter2017_cost_function(X, x, y_target, model, features)
    return watcher2017_weighted(x_prime, 0) # lambda=0 to get distance


'''
create individual with intrinsically bounded features.
to test other methods, simply change overwrite bounds, and weight functions
'''
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

    def num_weight_function():
        return get_constant_weight_function(random.uniform(0.1, 10)) if random.random() < 0.5 else None

    def obj_weight_function(value):
        return get_constant_category_weight_function(random.uniform(0.1, 10), value) if random.random() < 0.5 else None

    feature_info = [
        ('age', 0, 'increase'),
        ('workclass', 1, 'unique'),
        ('educational-num', 2, 'increase'),
        ('marital-status', 3, 'unique'),
        ('occupation', 4, 'unique'),
        ('relationship', 5, 'unique'),
        ('race', 6, 'fixed'),
        ('gender', 7, 'fixed'),
        ('capital-gain', 8, 'range'),
        ('capital-loss', 9, 'range'),
        ('hours-per-week', 10, 'range'),
        ('native-country', 11, 'fixed')
    ]

    for feature_name, index, boundry_type in feature_info:
        value = x.values[0][index]
        
        if boundry_type == 'unique':
            boundry = X[feature_name].unique()
            weight_function = obj_weight_function(value)
        elif boundry_type == 'increase':
            std = X[feature_name].std()
            lower_boundry = value
            upper_boundry = int(value + std)
            boundry = (lower_boundry, upper_boundry)
            weight_function = num_weight_function()
        elif boundry_type == 'fixed':
            boundry = [value]
            weight_function = obj_weight_function(value)
        elif boundry_type == 'range':
            std = X[feature_name].std()
            lower_boundry = value - std
            upper_boundry = value + std
            boundry = (lower_boundry, upper_boundry)
            weight_function = num_weight_function()

        features.append(Feature(feature_name, value, boundry, weight_function))
    
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

def standard(features, X):
    # for each feature
    # - set numerical featutures to min and max
    # - categorical to unique values
    for feature in features:
        if feature.variable_type == int or feature.variable_type == float:
            min_value = X[feature.name].min()
            max_value = X[feature.name].max()
            feature.boundaries = (min_value, max_value)
        else:
            feature.boundaries = X[feature.name].unique()
    return features

def domain_restriction(features, X, x):
    # for each feature
    # - set numerical featutures to value +- std
    # - categorical to unique values
    for feature in features:
        if feature.variable_type == int or feature.variable_type == float:
            std = X[feature.name].std()
            value = x[feature.name].values[0]
            feature.boundaries = (value-std, value+std)
        else:
            feature.boundaries = X[feature.name].unique()
    return features

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

    # change this to determine method and cost function (run each once to get results)
    methods = ("standard", "domain restriction", "intrinsic restriction", "weighted")
    method = methods[1]

    # create n random individuals
    individuals = 10
    print(f"Testing {individuals} individuals")
    for i in range(individuals):
        print(i+1, end=" ")
        x, y, y_target, features = generate_individual(X, model, verbose=False)

        # if method is standard make all domains min and max of dataset
        if method == "standard":
            features_changed = standard(copy.deepcopy(features), X)

        # if method is domain restriction, set domains to be within +- 1 std of x
        if method == "domain restriction":
            features_changed = domain_restriction(copy.deepcopy(features), X, x)

        # if method is intrinsic restriction, no need to change, weights will be ignored with default cost function
        if method == "intrinsic restriction":
            features_changed = features

        # if method is weighted, no need to change, but use cost function with weights
        if method == "weighted":
            features_changed = features
            cost_func = create_weighted_wachter2017_cost_function(X, x, y_target, model, features)
        else:
            cost_func = create_wachter2017_cost_function(X, x, y_target, model)

        # use changed features for generating counterfactuals
        CFS = get_counterfactuals(x, y_target, model, cost_func, features_changed, tol=0.1, optimization_steps=250) # can change tol and steps
        
        # get best counterfactual if there are any
        if len(CFS) == 0:
            print("No counterfactuals found for this individual, try increase tol or steps")
            continue

        # test all CFS on each metric. now using actual features
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
        # save result for notebook analysis
        folder_path = f"notebooks/results/{metrics_to_test[i].__name__}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = f"{folder_path}/{method}.npy"
        np.save(file_path, result)

if __name__ == "__main__":
    test()