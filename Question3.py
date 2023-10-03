#Student ID: 20838709

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

def get_RMSE(Y_desired, Y_actual):
    summation = 0
    for des, act in zip(Y_desired, Y_actual):
        summation += (des - act) ** 2
        
    summation /= len(Y_desired)
    
    summation = round(np.sqrt(summation), 2)
    
    return summation

rng = np.random.RandomState(seed=0)

train_x = np.linspace(0, 1, 10)
train_y = np.sin(train_x*2*np.pi) + rng.normal(0,0.1,size=10)

test_x = np.linspace(0, 1, 500)
test_y = np.sin(test_x*2*np.pi)

# A)
print("A)")

plt.scatter(train_x, train_y, label= "Training Data")
plt.scatter(test_x, test_y, label= "Testing Data")

train_x_matrix = generate_x_train_matrix(train_x, 1)

weights = get_least_sqaures_inv(train_x_matrix) * np.transpose(np.matrix(train_y))

regression_y = train_x_matrix * weights
    
plt.scatter(train_x, np.array(regression_y), label= "Very Accurate \nLinear Regression")
plt.legend()
plt.title("3A) Linear Regression Against Training and Testing Data")

plt.show()
rmse_regression_vs_training = get_RMSE(train_y, np.array(regression_y.T)[0])
# Only want every 50th item from test_y, since test_x covers same range as
# train_x, but with 50x greater resolution.
rmse_regression_vs_testing = get_RMSE(test_y[::50], np.array(regression_y.T)[0])

print("Weights: ", weights.T)
print("RMSE Regression vs Training: ", rmse_regression_vs_training)
print("RMSE Regression vs Testing: ", rmse_regression_vs_testing)

print()


# B)

print("B)")

plt.scatter(train_x, train_y, label= "Training Data")
plt.scatter(test_x, test_y, label= "Testing Data")

train_x_matrix = generate_x_train_matrix(train_x, 4)

weights = get_least_sqaures_inv(train_x_matrix) * np.transpose(np.matrix(train_y))

regressed_output = train_x_matrix * weights

plt.scatter(train_x, np.array(regressed_output))

polynomial_y = np.polyval(weights[::-1], test_x)
plt.scatter(test_x, polynomial_y, color="orange", label="Degree 4 Polynomial Regression")
plt.title("3B) Fourth Degree Polynomial Regression Against Testing and Training Data")
plt.legend()
plt.show()

print("Regression weights: ", weights.T)

rmse_regression_vs_training = get_RMSE(train_y, np.array(regressed_output.T)[0])
# Only want every 50th item from test_y, since test_x covers same range as
# train_x, but with 50x greater resolution.
rmse_regression_vs_testing = get_RMSE(test_y[::50], np.array(regressed_output.T)[0])

print("RMSE Regression vs Training: ", rmse_regression_vs_training)
print("RMSE Regression vs Testing: ", rmse_regression_vs_testing)

print()


# C)

print("C)")


# This is calculating the RMSE of the training 
RMSEs_against_training = []
RMSEs_against_testing = []
max_poly_deg = 15
degree_linsapce = np.linspace(0, max_poly_deg, max_poly_deg + 1).astype(int)

for poly_degree in degree_linsapce:
    
    plt.scatter(train_x, train_y, label= "Training Data", color="black")
    plt.scatter(test_x, test_y, label = "Testing Data", color="blue")
    
    train_x_matrix = generate_x_train_matrix(train_x, poly_degree)
    
    weights = get_least_sqaures_inv(train_x_matrix) * np.transpose(np.matrix(train_y))
    
    regressed_output = train_x_matrix * weights
    
    plt.scatter(train_x, np.array(regressed_output), color='blue')
    
    polynomial_y = np.polyval(weights[::-1], test_x)
    
    plt.plot(test_x, polynomial_y, color="orange",
             label = "Degree " + str(poly_degree) + " Poly Regression")
    
    original_points_regressed = np.polyval(weights[::-1], train_x)
    
    plt.scatter(train_x, original_points_regressed, color="orange", label="Given X points along regression")
    
    regressed_output_for_rmse = np.array(regressed_output.T)[0]
    
    rmse_regression_vs_training = get_RMSE(train_y, regressed_output_for_rmse)
    rmse_regression_vs_testing = get_RMSE(test_y, polynomial_y)

    RMSEs_against_training.append(rmse_regression_vs_training)
    RMSEs_against_testing.append(rmse_regression_vs_testing)

#    print("Degree " + str(poly_degree) + " Regression")
#    print("Weights: ", weights.T)
#    print("RMSE Regression vs Training: ", rmse_regression_vs_training)
#    print("RMSE Regression vs Testing: ", rmse_regression_vs_testing)
#    print()
    
    plt.title("3C) Polynomial Regression against Training and Testing Data")
    plt.xlabel("Stimulus")
    plt.ylabel("Response")
    plt.legend()
    plt.show()
    
print()



plt.scatter(degree_linsapce, RMSEs_against_training,
            label="RMSE Against Training Data")

plt.scatter(degree_linsapce, RMSEs_against_testing,
            label="RMSE Against Testing Data")
plt.legend()
plt.xlabel("Polynomial Degree")
plt.ylabel("RMSE")
plt.title("3C) RMSEs Against Training & Testing Data")
plt.show()


# D)

print("D)")

lambds = np.exp(np.linspace(-50,-1, 50))

RMSEs_against_training = []
RMSEs_against_testing = []

for regularizer in lambds:

    train_x_matrix = generate_x_train_matrix(train_x, 10)
    
    weights = get_regularized_inv(train_x_matrix, regularizer) * np.transpose(np.matrix(train_y))
    
    regressed_output = train_x_matrix * weights
    
    polynomial_y = np.polyval(weights[::-1], test_x)
    plt.plot(test_x, polynomial_y, label="Degree 10 Polynomial", color="orange")
    
    plt.scatter(train_x, train_y, label= "Training Data", color="black")
    plt.scatter(test_x, test_y, label = "Testing Data", color="blue")
    
    original_points_regressed = np.polyval(weights[::-1], train_x)
    plt.scatter(train_x, original_points_regressed, color="orange", 
                label="Original Points Regressed")
    
    plt.title("3D) Training, Testing, and Polyfit data with \n Degree 10 and Regularizer = " + str(regularizer))
    plt.legend()
    plt.xlabel("Stimulus")
    plt.ylabel("Response")
    plt.show()
    
    regressed_output_for_rmse = np.array(regressed_output.T)[0]
    
    rmse_regression_vs_training = get_RMSE(train_y, regressed_output_for_rmse)
    rmse_regression_vs_testing = get_RMSE(test_y, polynomial_y)

    RMSEs_against_training.append(rmse_regression_vs_training)
    RMSEs_against_testing.append(rmse_regression_vs_testing)

plt.semilogx(lambds, RMSEs_against_training,
            label="RMSE Against Training Data")

plt.semilogx(lambds, RMSEs_against_testing,
            label="RMSE Against Testing Data")
plt.legend()
plt.xlabel("Regularizer")
plt.ylabel("RMSE")
plt.title("3D) RMSEs Against Training & Testing Data")
plt.show()






