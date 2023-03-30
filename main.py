import pickle
import pandas as pd
from counterfactuals import get_counterfactuals
from cost_functions import wachter2017_cost_function, weighted_watcher_cost_function
import warnings
from prefrences import Feature, get_constant_weight_function, get_pow_weight_function
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.5f}'.format

def main():
    with open('pretrained_models/model_svc_diabetes.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('datasets/diabetes.csv')
    X = df.drop(columns = 'Outcome')
    y = df['Outcome']

    x_example = X.loc[0:0] 
    y_example = y.loc[0]
    
    P_prime = 0.1

    features = []
    features.append(Feature('Pregnancies', int, (x_example.values[0][0], 10), get_constant_weight_function(1)))
    features.append(Feature('Glucose', float, (100, 200), get_constant_weight_function(1)))
    features.append(Feature('BloodPressure', float, (50, 200), get_constant_weight_function(1)))
    features.append(Feature('SkinThickness', float, (20, 50), get_constant_weight_function(1)))
    features.append(Feature('Insulin', float, (0, 100), get_constant_weight_function(1)))
    features.append(Feature('BMI', int, (1, 100), get_constant_weight_function(1)))
    features.append(Feature('DiabetesPedigreeFunction', float, (0, 5), get_constant_weight_function(1)))
    features.append(Feature('Age', float, (x_example.values[0][-1], x_example.values[0][-1]+10), get_pow_weight_function(x_example.values[0][-1], x_example.values[0][-1]+10, 20, 1000)))
    # features.append(Feature('Age', float, (x_example.values[-1], x_example.values[-1]+10), get_constant_weight_function(1)))

    counterfactuals = get_counterfactuals(x_example, P_prime, model, X, 
                                          cost_function=weighted_watcher_cost_function, tol=0.1, 
                                          features=features, optimization_method="optuna", optimization_steps=20)

    print('Original example:')
#    print(x_example.to_frame().T)
    print(x_example)
    print(f"Predicted probability: {model.predict_proba(x_example.values.reshape((1, -1)))[0][0]}")
    print(f"Wanted probability: {P_prime}")
    print(f"Actual probability: {y_example}")
    print()
    print('Counterfactuals:')
    print(counterfactuals)
    print("Predicted probabilities:")
    for i in range(counterfactuals.shape[0]):
        print(model.predict_proba(counterfactuals.iloc[i].values.reshape((1, -1)))[0][0])
    print()
    print('Difference between original example and counterfactuals:')
    print(counterfactuals - x_example.values)
    print()

if __name__ == "__main__":
    main()
    # categories