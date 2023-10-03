# Student ID: 20838709

import sklearn.linear_model
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from sklearn import datasets

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

diabetes = datasets.load_diabetes()

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    diabetes.data, diabetes.target, test_size=0.3333, shuffle=True,
)

X_validate, X_train, Y_validate, Y_train = sklearn.model_selection.train_test_split(
    X_train, Y_train, test_size=0.5, shuffle=True,
)

lambds = np.exp(np.linspace(-20,5,50))

# A)
print("A) ")

training_rmses = []
validation_rmses = []

for regularizer in lambds:
    reg = sklearn.linear_model.Ridge(alpha = regularizer)
    
    reg.fit(X_train, Y_train)
    
    training_predictions = reg.predict(X_train)
    validation_predictions = reg.predict(X_validate)
    
    rms_train = rms_between_arrays(training_predictions, Y_train)
    rms_validation = rms_between_arrays(validation_predictions, Y_validate)
 
    training_rmses.append(rms_train)
    validation_rmses.append(rms_validation)

plt.xscale('log')
plt.scatter(lambds, training_rmses, label="Training Vs Fit, RMSE")
plt.scatter(lambds, validation_rmses, label="Validation Vs Fit, RMSE")

plt.xlabel("Regularizer Value")
plt.ylabel("RMSE")
plt.title("4A) RMSE vs Regularizer Value")
plt.ylim(0, 100)
plt.legend()
plt.show()

best_regularizer = lambds[validation_rmses.index(min(validation_rmses))]
print("Best Regularizer: ", best_regularizer)

reg = sklearn.linear_model.Ridge(alpha = best_regularizer)

reg.fit(X_train, Y_train)

testing_predictions = training_predictions = reg.predict(X_test)
rms_test = rms_between_arrays(testing_predictions, Y_test)

print("RMS for testing data: ", rms_test)


#B

# Running the code multiple times yields RMSE curves that vary 10-20% along the whole curve.
# However, the general shape of the curve stays the same from trial to trail.
# In most trials, the RMSE for both training and testing is only increasing with the regularizer.
# Overfitting would suggest that the regression has a much lower RMSE with the training data
# than it does with the testing data, but the RMSE for all regularizer values is not observed
# To be consistently higher for the training data than it is for the testing data.

# Even though the vertical offset of the RMSE for the training and validating 
# Changes from trial to trial, the best regularizer only moves around a little bit.

# C/D)
print("C/D) ")

training_rmses_per_deg = []
validating_rmses_per_deg = []
max_deg = 6

for poly_deg in range(0, max_deg):
    training_rmses = []
    validating_rmses = []
    print(poly_deg)
    i = 0
    for regularizer in lambds:
        F = np.matrix(sklearn.preprocessing.PolynomialFeatures(degree = poly_deg).fit_transform(X_train))
        regularized_inv = get_regularized_inv(F, regularizer)
        weights = regularized_inv * np.transpose(np.matrix(Y_train))
                
        regressed_training = np.array(weights.T * F.T)
        rms_polys_against_training = rms_between_arrays(regressed_training.T, Y_train)
        
        F_validate = np.matrix(sklearn.preprocessing.PolynomialFeatures(degree = poly_deg).fit_transform(X_validate))
        regressed_validating = np.array(weights.T * F_validate.T)
        rms_polys_against_validating = rms_between_arrays(regressed_validating.T, Y_validate)
         
        training_rmses.append(rms_polys_against_training)
        validating_rmses.append(rms_polys_against_validating)
        print(i)
        i+=1

    training_rmses_per_deg.append(training_rmses)
    validating_rmses_per_deg.append(validating_rmses)
    
for poly_deg in range(0, max_deg):
    plt.plot(lambds, training_rmses_per_deg[poly_deg], label="Poly Degree " + str(poly_deg))
    
plt.ylim(0, 125)
plt.xscale('log')
plt.xlabel("Regularizer")
plt.ylabel("RMSE")
plt.title("4C/D) Polynomial Fits' RMSEs against Regularizer Values, Training Data")    
plt.legend()
plt.grid()
plt.show()

for poly_deg in range(0, max_deg):
    plt.plot(lambds, validating_rmses_per_deg[poly_deg], label="Poly Fit Degree " + str(poly_deg))
    
plt.ylim(0, 125)
plt.xscale('log')
plt.xlabel("Regularizer")
plt.ylabel("RMSE")
plt.title("4C/D) Polynomial Fits' RMSEs against Regularizer Values, Validating Data")    
plt.legend()
plt.grid()
plt.show()



# My laptop can barely handle polynomials of degree 4, so I'll have to hypothesize
# That the trend observed from polynomials 1 - 4 will continue onwards to polynomials
# Of higher degrees. 





