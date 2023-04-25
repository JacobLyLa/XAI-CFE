import matplotlib.pyplot as plt
import numpy as np

"""
    Returns a function that always returns the same value
"""
def get_constant_category_weight_function(const_weight, category):
    def constant_category_weight_function(x):
        if x != category:
            return const_weight
        else:
            return 0
    return constant_category_weight_function

def get_constant_weight_function(const_weight):
    def constant_weight_function(x):
        return x*const_weight
    return constant_weight_function

def get_pow_weight_function(a, b, j, c):
    def pow_weight_function(x):
        if x < a or x > b:
            return c
        d = c/b**j * 1/(1/b**j - 1/a**j)
        k = -d/a**j
        return k*x**j + d
    return pow_weight_function

def plot_function(function, x_min, x_max):
    x = np.linspace(x_min, x_max, 100)
    y = [function(x_i) for x_i in x]
    plt.plot(x, y)
    plt.show()

"""
    Helper class for defining features values and weight functions.
    Samples a value between the given boundaries and close to initial_value for faster convergence.
"""
class Feature:
    def __init__(self, name, initial_value, variable_type, boundaries, weight_function=None):
        self.name = name
        self.initial_value = initial_value
        self.variable_type = variable_type
        self.boundaries = boundaries
        self.weight_function = weight_function
        
        if self.weight_function is None:
            if self.variable_type == int or self.variable_type == float:
                self.weight_function = get_constant_weight_function(1)
            else:
                self.weight_function = get_constant_category_weight_function(1, self.initial_value)

        self.value = None
        self.value_history = []

    def sample(self, trial):
        if self.variable_type == int:
            self.value = trial.suggest_int(self.name, self.boundaries[0], self.boundaries[1])
            self.value_history.append(self.value)
        elif self.variable_type == float:
            self.value = trial.suggest_float(self.name, self.boundaries[0], self.boundaries[1])
            self.value_history.append(self.value)
        elif self.variable_type == object:
            self.value = trial.suggest_categorical(self.name, self.boundaries)
            self.value_history.append(self.value)
        else:
            raise ValueError("variable_type must be int, float or object")

    def __str__(self):
        return f"{self.name}={self.value} boundaries={self.boundaries}"


if __name__ == "__main__":
    # define a feature
    feature = Feature("age", 30, int, (30, 100), get_pow_weight_function(50, 60, 20, 1000))
    # plot the weight function
    plot_function(feature.weight_function, 0, 100)