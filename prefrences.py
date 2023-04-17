import matplotlib.pyplot as plt
import numpy as np

"""
    Returns a function that always returns the same value
"""
def get_constant_weight_function(const_weight):
    def constant_weight_function(x):
        return const_weight
    return constant_weight_function

def get_line_slope_function(slope):
    def constant_line_slope_function(x):
        return x*slope
    return constant_line_slope_function

def get_pow_weight_function(a, b, j, c):
    a += (b - a)/2
    def pow_weight_function(x):
        if x < a:
            return pow_weight_function(2*a-x)
        if x > b:
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
    Helper class for defining features values and weights
"""
class Feature:
    def __init__(self, name, variable_type, boundaries, weight_function=None):
        self.name = name
        self.variable_type = variable_type
        self.boundaries = boundaries
        self.weight_function = weight_function
        if self.weight_function == None:
            if self.variable_type == int or self.variable_type == float:
                self.weight_function = get_constant_weight_function(1)

        self.value = None

    def sample(self, trial):
        if self.variable_type == int:
            self.value = trial.suggest_int(self.name, self.boundaries[0], self.boundaries[1])
        elif self.variable_type == float:
            self.value = trial.suggest_uniform(self.name, self.boundaries[0], self.boundaries[1])
        else:
            self.value = trial.suggest_categorical(self.name, self.boundaries)

    def __str__(self):
        return f"{self.name}: boundaries: {self.boundaries}"

if __name__ == "__main__":
    # define a feature
    feature = Feature("age", int, (30, 100), get_pow_weight_function(5,35, 15, 1000))
    # plot the weight function
    plot_function(feature.weight_function, 0, 100)