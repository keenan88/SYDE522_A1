import sklearn.datasets
import sklearn.linear_model
import sklearn.model_selection
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

get_ipython().magic('clear')
get_ipython().magic('reset -f')

digits = sklearn.datasets.load_digits()


"""
plt.figure(figsize=(12,3))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(digits.data[i].reshape(8,8), cmap='gray_r')
    plt.title(f'target:{digits.target[i]}')
plt.show()
"""

# Section A
print("A)")
digits_0_1 = digits.data[(digits.target == 0) | (digits.target == 1)]
digits_0_1_classified = digits.target[(digits.target == 0) | (digits.target == 1)]

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    digits_0_1, digits_0_1_classified, test_size=0.2, shuffle=True,
)

perceptron = sklearn.linear_model.Perceptron(eta0 = 1.0)

perceptron.fit(X_train, Y_train)

predictions_0_1 = perceptron.predict(X_test)

correct_predictions = 0

for prediction, actual in zip(predictions_0_1, Y_test):
    if prediction == actual:
        correct_predictions += 1
        
pct_accuracy = round(100 * correct_predictions / len(Y_test), 2)

print("Percent Accuracy: ", pct_accuracy, "%")
print()


# Section B
print("B)")

xmax = 10000
learning_rates = np.linspace(0.1, xmax, 25)
learning_rates = [1]
iterations_per_learning_rate = 10
#learning_rates = [1]
accuracies = []
misclassifications = [[0] * 10 for _ in range(10)]
misclassifications = np.zeros((10, 10))

for learning_rate in learning_rates:
    
    learning_rates_accuracies = []
    
    perceptron = sklearn.linear_model.Perceptron(eta0 = learning_rate)
    
    for _ in range(iterations_per_learning_rate):    

        X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
            digits.data, digits.target, test_size=0.2, shuffle=True,
        )
        
        perceptron.fit(X_train, Y_train)
        
        predictions = perceptron.predict(X_test)
        
        correct_predictions = 0
        
        for prediction, actual in zip(predictions, Y_test):
            if prediction == actual:
                correct_predictions += 1
            else:
                misclassifications[actual][prediction] += 1
                
        pct_accuracy = round(100 * correct_predictions / len(Y_test), 2)
        learning_rates_accuracies.append(pct_accuracy)
        
    avgd_accuracy_for_learning_rate = sum(learning_rates_accuracies) / len(learning_rates_accuracies)
    accuracies.append(avgd_accuracy_for_learning_rate)
        
        


plt.figure(figsize=(6,6))
plt.scatter(learning_rates, accuracies)
plt.xlabel('Learning Rate')
plt.ylabel('% Accuracy')

plt.xlim(0, xmax)
plt.ylim(0, 100)

plt.show()

print()


#C
print("C) ")

flat_matrix = misclassifications.flatten()






















