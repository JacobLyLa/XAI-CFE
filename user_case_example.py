import math
import pickle
import warnings

import pandas as pd

from cost_functions import (create_wachter2017_cost_function,
                            create_weighted_wachter2017_cost_function)
from counterfactuals import get_counterfactuals
from prefrences import (Feature, get_constant_category_weight_function,
                        get_constant_weight_function, get_pow_weight_function)

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.4f}'.format


def main():
    with open('pretrained_models/pipeline_adult.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('datasets/adult.csv', na_values='?')
    df = df.drop(columns = ['fnlwgt'])
    df = df.dropna()
    X = df.drop(columns=['income', 'education'])

    ashley = {'age': 30,
          'workclass': 'Private',
          'educational-num': 10,
          'marital-status': 'Never-married',
          'occupation': 'Sales', # so she is a cashier, or independent consultant for Oriflame
          'relationship': 'Own-child',
          'race': 'White',
          'gender': 'Female',
          'capital-gain': 0,
          'capital-loss': 0,
          'hours-per-week': 25,
          'native-country': 'United-States'}
    ashley_df = pd.DataFrame(ashley, index=[0])
    ashley_values = ashley_df.values[0]
    print("Ashley df:")
    print(ashley_df)
    
    print()
    features = []
    features.append(Feature('age', ashley_values[0], (30, 50), get_constant_weight_function(1)))
    features.append(Feature('workclass', ashley_values[1], X['workclass'].unique()))
    features.append(Feature('educational-num', ashley_values[2], (10, 14), get_constant_weight_function(1)))
    features.append(Feature('marital-status', ashley_values[3], X['marital-status'].unique()))
    features.append(Feature('occupation', ashley_values[4], X['occupation'].unique()))
    features.append(Feature('relationship', ashley_values[5], X['relationship'].unique()))
    features.append(Feature('race', ashley_values[6], X['race'].unique(), get_constant_category_weight_function(10.0, ashley_values[6])))
    features.append(Feature('gender', ashley_values[7], X['gender'].unique(), get_constant_category_weight_function(1, ashley_values[7])))
    features.append(Feature('capital-gain', ashley_values[8], (0, 10000), get_constant_weight_function(1))) # too high here can give very high cost
    features.append(Feature('capital-loss', ashley_values[9], (0, 1000), get_constant_weight_function(1))) # and here
    features.append(Feature('hours-per-week', ashley_values[10], (0, 70), get_constant_weight_function(1)))
    features.append(Feature('native-country', ashley_values[11], X['native-country'].unique()))

    normal_watcher = create_wachter2017_cost_function(X=X, x=ashley_df, y_target=0.3, model=model)
    weighted_watcher = create_weighted_wachter2017_cost_function(X=X, x=ashley_df, y_target=0.3, model=model, features=features)

    normal_CFS = get_counterfactuals(x=ashley_df, y_target=0.3, model=model, cost_function=normal_watcher, features=features, tol=0.1, optimization_steps=100)
    # weighted_CFS = get_counterfactuals(x=ashley_df, y_target=0.3, model=model, cost_function=weighted_watcher, features=features, tol=0.1, optimization_steps=100)
    print(model.predict_proba(ashley_df))
    print("Ashley Counterfactuals:")
    print(normal_CFS)
    print(model.predict_proba(normal_CFS))
    

if __name__ == "__main__":
    main()