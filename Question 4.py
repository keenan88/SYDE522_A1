import sklearn.linear_model
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt

get_ipython().magic('clear')
get_ipython().magic('reset -f')

def rms_between_arrays(array1, array2):
    squared_diff = (array1 - array2) ** 2
    mean_squared_diff = np.mean(squared_diff)
    rms = np.sqrt(mean_squared_diff)

    return rms

def get_regularized_inv(train_x_matrix, regularizer):
    
    comp1 = np.transpose(train_x_matrix) * train_x_matrix
    comp2 = comp1 + regularizer * np.identity(comp1.shape[0])
    comp2 = np.linalg.pinv(comp2)
    regularized_inv = comp2 * np.transpose(train_x_matrix)
    
    return regularized_inv

diabetes = sklearn.datasets.load_diabetes()

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    diabetes.data, diabetes.target, test_size=0.3333, shuffle=True,
)

X_validate, X_train, Y_validate, Y_train = sklearn.model_selection.train_test_split(
    X_train, Y_train, test_size=0.5, shuffle=True,
)
lambds = np.exp(np.linspace(-20,5,50))

# A)
print("A) ")

"""

training_rmses = []
validation_rmses = []

for regularizer in lambds:
    reg = sklearn.linear_model.Ridge(alpha = regularizer)
    
    reg.fit(diabetes.data, diabetes.target)
    
    
    training_predictions = reg.predict(X_train)
    validation_predictions = reg.predict(X_validate)
    
    rms_train = rms_between_arrays(training_predictions, Y_train)
    rms_validation = rms_between_arrays(validation_predictions, Y_validate)
 
    training_rmses.append(rms_train)
    validation_rmses.append(rms_validation)

plt.xscale('log')
plt.scatter(lambds, training_rmses)
plt.scatter(lambds, validation_rmses)

plt.ylim(0, 100)

plt.show()
"""


# C)
print("C) ")
poly_rmses = []

F = np.matrix(sklearn.preprocessing.PolynomialFeatures(degree=2).fit_transform(diabetes.data))

for regularizer in lambds:
    regularized_inv = get_regularized_inv(F, regularizer)
    
    weights = regularized_inv * np.transpose(np.matrix(diabetes.target))
    
    regressed_output = np.array(F * weights)
    
    rms_polys = rms_between_arrays(regressed_output, diabetes.target)
     
    poly_rmses.append(rms_polys)

print(poly_rmses)

plt.xscale('log')
plt.scatter(lambds, poly_rmses)

plt.ylim(0, 100)

plt.show()






