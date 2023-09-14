



import matplotlib.pyplot as plt
import numpy as np
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

def get_regularized_inv(train_x_matrix, regularizer):
    
    comp1 = np.transpose(train_x_matrix) * train_x_matrix
    comp2 = comp1 + regularizer * np.identity(comp1.shape[0])
    comp2 = np.linalg.pinv(comp2)
    regularized_inv = comp2 * np.transpose(train_x_matrix)
    
    return regularized_inv

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
4
polynomial_y = np.polyval(weights[::-1], precise_x)
plt.plot(precise_x, polynomial_y, label="Degree 4 Polynomial", color="orange")
plt.show()

print()


# C)

print("C)")

for poly_degree in range(0,15):

    plt.scatter(train_x, train_y)
    
    train_x_matrix = generate_x_train_matrix(train_x, poly_degree)
    
    weights = get_least_sqaures_inv(train_x_matrix) * np.transpose(np.matrix(train_y))
    
    regressed_output = train_x_matrix * weights
    
    plt.scatter(train_x, np.array(regressed_output), color='blue')
    
    polynomial_y = np.polyval(weights[::-1], precise_x)
    plt.plot(precise_x, polynomial_y, label="Degree 4 Polynomial", color="orange")
    
    original_points_regressed = np.polyval(weights[::-1], train_x)
    plt.scatter(train_x, original_points_regressed, color="orange")
    
    plt.show()
    
print()
"""

# D)

print("D)")




lambds = np.exp(np.linspace(-50,-1, 50))

for regularizer in lambds:

    train_x_matrix = generate_x_train_matrix(train_x, 10)
    
    weights = get_regularized_inv(train_x_matrix, regularizer) * np.transpose(np.matrix(train_y))
    
    regressed_output = train_x_matrix * weights
    
    plt.scatter(train_x, np.array(regressed_output), color='blue')
    
    polynomial_y = np.polyval(weights[::-1], precise_x)
    plt.plot(precise_x, polynomial_y, label="Degree 4 Polynomial", color="orange")
    
    plt.scatter(train_x, train_y, color='blue')
    
    original_points_regressed = np.polyval(weights[::-1], train_x)
    plt.scatter(train_x, original_points_regressed, color="orange")
    
    plt.show()
        
    print()









