



import matplotlib.pyplot as plt
import numpy as np
import copy
from IPython import get_ipython

get_ipython().magic('clear')
get_ipython().magic('reset -f')

def generate_x_train_matrix(x_test_points, highest_degree_poly):

    columns = []

    for degree in range(highest_degree_poly + 1):
        columns.append(train_x ** degree)

    train_x_matrix = np.matrix(np.column_stack(columns))
    
    return train_x_matrix
    
def get_least_sqaures_inv(train_x_matrix):
    comp1 = np.transpose(train_x_matrix) * train_x_matrix
    comp2 = np.linalg.pinv(comp1)
    least_sqaures_inv = comp2 * np.transpose(train_x_matrix)

    return least_sqaures_inv

rng = np.random.RandomState(seed=0)
train_x = np.linspace(0, 1, 10)
precise_x = np.linspace(0, 1, 100)
train_y = np.sin(train_x*2*np.pi) + rng.normal(0,0.1,size=10)
"""
# A)
print("A)")

plt.scatter(train_x, train_y)

train_x_matrix = generate_x_train_matrix(train_x, 1)

weights = get_least_sqaures_inv(train_x_matrix) * np.transpose(np.matrix(train_y))

regression_y = train_x_matrix * weights
    
plt.scatter(train_x, np.array(regression_y))

plt.show()

print()


# B)

print("B)")

plt.scatter(train_x, train_y)

train_x_matrix = generate_x_train_matrix(train_x, 4)

weights = get_least_sqaures_inv(train_x_matrix) * np.transpose(np.matrix(train_y))

regressed_output = train_x_matrix * weights

plt.scatter(train_x, np.array(regressed_output))

polynomial_y = np.polyval(weights[::-1], precise_x)
plt.plot(precise_x, polynomial_y, label="Degree 4 Polynomial", color="orange")
plt.show()

print()
"""

# C)

print("C)")

plt.scatter(train_x, train_y)

train_x_matrix = generate_x_train_matrix(train_x, 10)

weights = get_least_sqaures_inv(train_x_matrix) * np.transpose(np.matrix(train_y))

regressed_output = train_x_matrix * weights

plt.scatter(train_x, np.array(regressed_output))

polynomial_y = np.polyval(weights[::-1], precise_x)
plt.plot(precise_x, polynomial_y, label="Degree 4 Polynomial", color="orange")
plt.show()

print()









