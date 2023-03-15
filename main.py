import pickle
import pandas as pd
from counterfactuals import get_counterfactuals
from cost_functions import wachter2017_cost_function
import warnings
warnings.filterwarnings('ignore')

def main():
    with open('pretrained_models/model_svc_diabetes.pkl', 'rb') as f:
        model = pickle.load(f)

    df = pd.read_csv('datasets/diabetes.csv')
    X = df.drop(columns = 'Outcome')
    y = df['Outcome']
    x_example = X.loc[0]
    y_example = y.loc[0]
    
    P_prime = 0.3
    boundaries = [(int, x_example.values[0], 10), (float, 100,200), (float, 50, 200), 
                  (float, 20, 50), (float, 0, 100), (int, 1, 100), 
                  (float, 0, 5), (int, x_example.values[-1], 100)] 
    counterfactuals = get_counterfactuals(x_example.values, P_prime, model, X, 
                                          cost_function=wachter2017_cost_function, tol=0.03, 
                                          boundaries=boundaries, optimization_method="scipy", optimization_steps=10)
    
    print('Original example:')
    print(x_example.to_frame().T)
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