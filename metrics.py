import numpy as np
import pickle
import pandas as pd
from counterfactuals import get_counterfactuals
import random
from cost_functions import wachter2017_cost_function, weighted_watcher_cost_function, wachter2017_cost_function_with_categorical, wachter2017_cost_function_ignore_categorical
import warnings
from prefrences import Feature, get_constant_weight_function, get_pow_weight_function
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.5f}'.format
np.random.seed(42)
random.seed(42)

def test_metric(x_list, p_list, p_prime_list, features_list, X, model, metric, cost_func, steps):
    results = []
    for i in range(len(x_list)):
        # get the counterfactual
        counterfactuals = get_counterfactuals(x_list[i], p_prime_list[i], model, X, 
                                            cost_function=cost_func, tol=0.2, 
                                            features=features_list[i], optimization_method="optuna", optimization_steps=steps)
        
        # get best counterfactual if there are any
        # TODO: maybe a metric that checks how many time it fails generating any counterfactuals
        if len(counterfactuals) == 0:
            continue

        # best CF
        x_prime = counterfactuals.iloc[0]
        results.append(metric(x_list[i].iloc[0], x_prime, X, features_list[i]))
    return np.array(results)

'''
Takes the original x and x_prime and returns the amount of features that have been changed
'''
def sparsity(x, x_prime, X, features):
    # count the number of features that have been changed
    count = 0
    for i in range(len(x)):
        if x[i] != x_prime[i]:
            count += 1
    return count / len(x)

'''
Takes the original x and x_prime and returns the amount of features that are off boundry
'''
def off_boundry(x, x_prime, X, features):
    # count how many features that are off boundry
    off_boundry = 0
    for i in range(len(x)):
        if x_prime[i] < features[i].boundaries[0] or x_prime[i] > features[i].boundaries[1]:
            off_boundry += 1
    return off_boundry / len(x)

'''
Takes the original x and x_prime and returns the closest datapoint in X
'''
def distance_closest(x, x_prime, X, features):
    # find candidates in X where the str columns are the same as in x_prime
    candidates = X.copy()
    for i in range(len(x_prime)):
        if features[i].variable_type == object:
            candidates = candidates[candidates[features[i].name] == x_prime[i]]
    # remove the str columns
    candidates = candidates.drop(columns=[features[i].name for i in range(len(x_prime)) if features[i].variable_type == object])
    # also remove them in x prime
    x_prime = x_prime.drop(index=[features[i].name for i in range(len(x_prime)) if features[i].variable_type == object])
    # calculate the distance between x_prime and the closest datapoint in X
    if len(candidates) == 0:
        return 1000000

    return np.linalg.norm(x_prime - candidates, axis=1).min()

def total_change(x, x_prime, X, features):
    # copy x and x_prime
    x = x.copy()
    x_prime = x_prime.copy()
    # weight the features
    for i in range(len(x)):
        if features[i].variable_type == object:
            # 1 to both if they are the same
            if x[i] == x_prime[i]:
                x[i] = 1
                x_prime[i] = 1
            else:
                x[i] = 0
                x_prime[i] = features[i].weight_function(1)

    return np.linalg.norm(x_prime - x, ord=2)

def total_change_weighted(x, x_prime, X, features):
    # copy x and x_prime
    x = x.copy()
    x_prime = x_prime.copy()
    # weight the features
    for i in range(len(x)):
        # if feature is categorical then skip
        if features[i].variable_type != object:
            x[i] = features[i].weight_function(x[i])
            x_prime[i] = features[i].weight_function(x_prime[i])
        else:
            # 1 to both if they are the same
            if x[i] == x_prime[i]:
                x[i] = 1
                x_prime[i] = 1
            else:
                x[i] = 0
                x_prime[i] = features[i].weight_function(1)

    return np.linalg.norm(x_prime - x, ord=2)

def generate_individual(X, model, visualize=False):
    x_index = random.randint(0, len(X)-1)
    x = X.loc[x_index:x_index]
    # calculate p, from the model
    p = model.predict_proba(x)[0][0]
    # adjust p to +- 0.2 probabilisticly, and make sure it is between 0 and 1
    p_prime = p + random.uniform(-0.3, 0.3)
    p_prime = max(0.0, min(1.0, p_prime))

    # create personalized features
    features = []

    feature_name = 'age'
    lower_boundry = int(x[feature_name])
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, int, (lower_boundry, upper_boundry), get_constant_weight_function(random.uniform(0.2, 3))))

    feature_name = 'workclass'
    features.append(Feature(feature_name, object, X[feature_name].unique(), get_constant_weight_function(random.uniform(0.2, 3))))

    feature_name = 'educational-num'
    lower_boundry = int(x[feature_name])
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, int, (lower_boundry, upper_boundry), get_constant_weight_function(random.uniform(0.2, 3))))

    feature_name = 'marital-status'
    features.append(Feature(feature_name, object, X[feature_name].unique(), get_constant_weight_function(random.uniform(0.2, 3))))
    
    feature_name = 'occupation'
    features.append(Feature(feature_name, object, X[feature_name].unique(), get_constant_weight_function(random.uniform(0.2, 3))))

    feature_name = 'relationship'
    features.append(Feature(feature_name, object, X[feature_name].unique(), get_constant_weight_function(random.uniform(0.2, 3))))

    feature_name = 'race'
    features.append(Feature(feature_name, object, X[feature_name].unique(), get_constant_weight_function(random.uniform(0.2, 3))))

    feature_name = 'gender'
    features.append(Feature(feature_name, object, X[feature_name].unique(), get_constant_weight_function(random.uniform(0.2, 3))))

    feature_name = 'capital-gain'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, int, (lower_boundry, upper_boundry), get_constant_weight_function(random.uniform(0.2, 3))))

    feature_name = 'capital-loss'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, int, (lower_boundry, upper_boundry), get_constant_weight_function(random.uniform(0.2, 2))))

    feature_name = 'hours-per-week'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, int, (lower_boundry, upper_boundry), get_constant_weight_function(random.uniform(0.2, 2))))

    feature_name = 'native-country'
    features.append(Feature(feature_name, object, X[feature_name].unique(), get_constant_weight_function(random.uniform(0.2, 2))))
    
    # visualize
    if visualize:
        print('x:')
        print(x)
        print()
        print('p:', p)
        print('p_prime:', p_prime)
        print('features:')
        print("-"*50)
        for feature in features:
            print(feature)
        print("-"*50)

    return x, p, p_prime, features

def test():
    with open('pretrained_models/pipeline_adult.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('datasets/adult.csv', na_values='?')
    df = df.drop(columns = ['fnlwgt'])
    df = df.dropna()
    # fix indices
    df = df.reset_index(drop=True)
    X =  df.drop(columns=['income', 'education'])
    y = df['income']

    metric_to_test = [sparsity, distance_closest, total_change, total_change_weighted]
    fail_count = 0
    individuals = 10
    for metric in metric_to_test:
        x_list = []
        p_list = []
        p_prime_list = []
        features_list = []
        # create n random examples
        for i in range(individuals):
            x, p, p_prime, features = generate_individual(X, model, visualize=False)
            x_list.append(x)
            p_list.append(p)
            p_prime_list.append(p_prime)
            features_list.append(features)
        results = test_metric(x_list, p_list, p_prime_list, features_list, X, model, metric, wachter2017_cost_function_with_categorical, 500)
        fail_count += individuals - len(results)
        print("Results for metric:", metric.__name__)
        print(results)
        print("Average:", results.mean())
        print("Standard deviation:", results.std())
        print("Median:", np.median(results))
        print("-"*40)
    print("Failed individuals:", fail_count)
    print("Chance of failure:", fail_count/(len(metric_to_test)*individuals))

    
if __name__ == "__main__":
    test()