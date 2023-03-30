import pickle
import pandas as pd
from counterfactuals import get_counterfactuals
from cost_functions import wachter2017_cost_function, weighted_watcher_cost_function, wachter2017_cost_function_ignore_categorical
import warnings
import math
from prefrences import Feature, get_constant_weight_function, get_pow_weight_function
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
pd.options.display.float_format = '{:.5f}'.format


def main():
    with open('pretrained_models/pipeline_adult.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('datasets/adult.csv', na_values='?')
    df = df.dropna()

    num_ix = df.select_dtypes(include=['int64', 'float64']).columns

    X = df.drop(columns=['income'])
    x=X.loc[15].values
    x = x.reshape((1, -1))
    sample_df = X.loc[12:12]
    print(sample_df)
    #print(model.predict_proba(sample_df))
    #print(model.predict_proba(sample_df))

  #  cost = wachter2017_cost_function_ignore_categorical(x_prime=sample_df, x=sample_df, y_prime=0.9, lambda_value=1, model = model, X=X)
   # print(cost)
    features = []
    features.append(Feature('age', float, (sample_df.values[0][0]-20, sample_df.values[0][0]+20), get_constant_weight_function(1)))
    features.append(Feature('fnlwgt', float, ( sample_df.values[0][2]-10, sample_df.values[0][2]+10), get_constant_weight_function(1)))
    features.append(Feature('educational-num', int, (sample_df.values[0][4], sample_df.values[0][4]+20), get_constant_weight_function(1)))
    features.append(Feature('capital-gain', int, (sample_df.values[0][10]-20, sample_df.values[0][10]+20), get_constant_weight_function(1)))
    features.append(Feature('capital-loss', int, (sample_df.values[0][11]-20, sample_df.values[0][11]+20), get_constant_weight_function(1)))
    features.append(Feature('hours-per-week', int, (sample_df.values[0][12], sample_df.values[0][12]+200), get_constant_weight_function(1)))
    counterfactuals = get_counterfactuals(x=sample_df, y_prime_target=0.7, model=model, X=X, cost_function=wachter2017_cost_function_ignore_categorical, \
                                           features=features, tol=0.2, optimization_method="optuna", optimization_steps=100, framed=True)

    print(counterfactuals)
    

if __name__ == "__main__":
    main()
    # categories