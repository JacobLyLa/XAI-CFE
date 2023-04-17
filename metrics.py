import numpy as np
import pickle
import pandas as pd
from counterfactuals import get_counterfactuals
import random
from cost_functions import wachter2017_cost_function, weighted_watcher_cost_function
import warnings
from prefrences import Feature, get_constant_weight_function, get_pow_weight_function
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.5f}'.format
# numpy set seed
np.random.seed(42)
# python set seed
random.seed(42)

def metric_statistics(x_list, p_list, p_prime_list, features_list, X, model):
    # standardize X, x and x_prime
    mean = X.mean()
    std = X.std()
    X_s = (X - mean) / std

    # only test 1 metric so far
    metric = total_change_weighted
    results = []
    for i in range(len(x_list)):
        # get the counterfactual
        counterfactuals = get_counterfactuals(x_list[i].values, p_prime_list[i], model, X, 
                                            cost_function=weighted_watcher_cost_function, tol=0.1, 
                                            features=features_list[i], optimization_method="scipy", optimization_steps=40)
        
        # get best counterfactual if there are any
        # TODO: maybe a metric that checks how many time it fails generating any counterfactuals
        if len(counterfactuals) == 0:
            continue

        x_prime = counterfactuals.loc[0]

        # make x copy
        x = x_list[i].copy()
        # standardize x
        x_s = (x - mean) / std
        # standardize x_prime
        x_prime_s = (x_prime - mean) / std

        results.append(metric(x_s.values, x_prime_s.values, features_list[i]))
    return np.array(results)

'''
Takes the original x and x_prime and returns the amount of features that have been changed
'''
def sparsity(x, x_prime):
    diff = x - x_prime
    diff = np.round(diff, 5)
    # count the number of features that have been changed
    return np.count_nonzero(diff) / len(x)

'''
Takes the original x and x_prime and returns the amount of features that are off boundry
'''
def off_boundry(x, x_prime, features):
    # count how many features that are off boundry
    off_boundry = 0
    for i in range(len(x)):
        if x_prime[i] < features[i].boundaries[0] or x_prime[i] > features[i].boundaries[1]:
            off_boundry += 1
    return off_boundry / len(x)

'''
Takes the original x and x_prime and returns the closest datapoint in X
'''
def distance_closest(x, x_prime, X):
    # calculate the distance between x_prime and the closest datapoint in X
    return np.linalg.norm(x_prime - X, axis=1).min()

'''

'''
def total_change(x, x_prime):
    return np.linalg.norm(x_prime - x, ord=2)

def total_change_weighted(x, x_prime, features):
    # copy x and x_prime
    x = x.copy()
    x_prime = x_prime.copy()
    # weight the features
    for i in range(len(x)):
        x[i] *= features[i].weight_function(x[i])
        x_prime[i] *= features[i].weight_function(x_prime[i])

    return np.linalg.norm(x_prime - x, ord=2)

def generate_individual(X, model, visualize=False):
    x_index = random.randint(0, len(X)-1)
    x = X.loc[x_index]
    # calculate p, from the model
    p = model.predict_proba(x.values.reshape((1, -1)))[0][0]
    # adjust p to +- 0.2 probabilisticly, and make sure it is between 0 and 1
    p_prime = p + random.uniform(-0.3, 0.3)
    p_prime = max(0.0, min(1.0, p_prime))

    # create personalized features
    features = []

    feature_name = 'Pregnancies'
    lower_boundry = int(x[feature_name])
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, int, (lower_boundry, upper_boundry), get_constant_weight_function(1)))

    feature_name = 'Glucose'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, float, (lower_boundry, upper_boundry), get_constant_weight_function(1)))

    feature_name = 'BloodPressure'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, float, (lower_boundry, upper_boundry), get_constant_weight_function(1)))

    feature_name = 'SkinThickness'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, float, (lower_boundry, upper_boundry), get_constant_weight_function(1)))
    
    feature_name = 'Insulin'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, float, (lower_boundry, upper_boundry), get_constant_weight_function(1)))

    feature_name = 'BMI'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, float, (lower_boundry, upper_boundry), get_constant_weight_function(1)))

    feature_name = 'DiabetesPedigreeFunction'
    lower_boundry = X[feature_name].min()
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, float, (lower_boundry, upper_boundry), get_constant_weight_function(1)))
    
    feature_name = 'Age'
    lower_boundry = int(x[feature_name])
    upper_boundry = X[feature_name].max()
    features.append(Feature(feature_name, int, (lower_boundry, upper_boundry), get_constant_weight_function(1)))
    
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
    with open('pretrained_models/model_svc_diabetes.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('datasets/diabetes.csv')
    X = df.drop(columns = 'Outcome')
    y = df['Outcome']

    '''
    x_example = X.loc[0]
    y_example = y.loc[0]
    
    P_prime = 0.1

    features = []
    features.append(Feature('Pregnancies', int, (x_example.values[0], 10), get_constant_weight_function(1)))
    features.append(Feature('Glucose', float, (100, 200), get_constant_weight_function(1)))
    features.append(Feature('BloodPressure', float, (50, 200), get_constant_weight_function(1)))
    features.append(Feature('SkinThickness', float, (20, 50), get_constant_weight_function(1)))
    features.append(Feature('Insulin', float, (0, 100), get_constant_weight_function(1)))
    features.append(Feature('BMI', int, (1, 100), get_constant_weight_function(1)))
    features.append(Feature('DiabetesPedigreeFunction', float, (0, 5), get_constant_weight_function(1)))
    features.append(Feature('Age', float, (x_example.values[-1], x_example.values[-1]+10), get_pow_weight_function(x_example.values[-1], x_example.values[-1]+10, 20, 1000)))

    # cost_function = weighted_watcher_cost_function
    cost_function = wachter2017_cost_function
    counterfactuals = get_counterfactuals(x_example.values, P_prime, model, X, 
                                          cost_function=cost_function, tol=0.1, 
                                          features=features, optimization_method="optuna", optimization_steps=10)
    
    counterfactual = counterfactuals.iloc[0]
    # standardize X, x and x_prime
    mean = X.mean()
    std = X.std()
    X_s = (X - mean) / std
    x_example_s = (x_example - mean) / std
    counterfactual_s = (counterfactual - mean) / std

    print('Original example:')
    print(x_example.to_frame().T)
    print(f"Predicted probability: {model.predict_proba(x_example.values.reshape((1, -1)))[0][0]}")
    print(f"Wanted probability: {P_prime}")
    print(f"Actual probability: {y_example}")
    print()
    print('Counterfactual:')
    print(counterfactual)
    print("Predicted probability: {}".format(model.predict_proba(counterfactual.values.reshape((1, -1)))[0][0]))
    print()
    print('Difference between original example and counterfactual:')
    print(counterfactual - x_example.values)
    print()
    print("Metrics for the counterfactual:")
    print("Sparsity: {}".format(sparsity(x_example_s.values, counterfactual_s.values)))
    print("Off boundry: {}".format(off_boundry(x_example.values, counterfactual.values, features)))
    print("Distance to closest datapoint: {}".format(distance_closest(x_example_s.values, counterfactual_s.values, X_s.values)))
    print("Total change: {}".format(total_change(x_example_s.values, counterfactual_s.values)))
    print("Total change weighted: {}".format(total_change_weighted(x_example_s.values, counterfactual_s.values, features)))
    '''

    # create n random examples
    x_list = []
    p_list = []
    p_prime_list = []
    features_list = []
    for i in range(10):
        x, p, p_prime, features = generate_individual(X, model, visualize=False)
        x_list.append(x)
        p_list.append(p)
        p_prime_list.append(p_prime)
        features_list.append(features)
    results = metric_statistics(x_list, p_list, p_prime_list, features_list, X, model)
    print(results)
    print("Average:", results.mean())
    print("Standard deviation:", results.std())
    print("Median:", np.median(results))

    
if __name__ == "__main__":
    test()