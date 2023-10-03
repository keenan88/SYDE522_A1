# Student ID: 20838709

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

# Part A [DONE]
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


# Part B [NEED REVIEW]
print("B)")

xmax = 10
#learning_rates = np.linspace(0.001, xmax, xmax)
learning_rates = np.logspace(start = -6, stop = 4, num = 100)
accuracies = []
num_misclassifications = np.zeros((10, 10))

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
    digits.data, digits.target, test_size=0.2, shuffle=True,
)

for learning_rate in learning_rates:
    
#    print("rate: ", learning_rate)
    
    perceptron = sklearn.linear_model.Perceptron(eta0 = learning_rate)

    perceptron.fit(X_train, Y_train)
    
    predictions = perceptron.predict(X_test)
    
    correct_predictions = 0
    
    for prediction, actual in zip(predictions, Y_test):
        if prediction == actual:
            correct_predictions += 1
        else:
            num_misclassifications[actual][prediction] += 1
            
    pct_accuracy = round(100 * correct_predictions / len(Y_test), 2)
#    print("Learning Rate: ", round(learning_rate, 2), ". Accuracy: ", pct_accuracy)
    
    accuracies.append(pct_accuracy)
        
        


plt.figure(figsize=(6,6))
plt.semilogx(learning_rates, accuracies)
plt.xlabel('Learning Rate')
plt.ylabel('% Accuracy')

plt.xlim(0, xmax)
#plt.ylim(90, 100)

plt.title("2B) Accuracy Vs Learning Rate")
plt.show()

print()


#C [DONE]
print("C) ")

flat_matrix = num_misclassifications.flatten()

# From observation, the most common misclassifications are 
# 8's being classified as 1's
# 1's being classified as 8's
# 9's being classified as 8's
# 3's being classified as 8's

def extract_every_10th_digit(offset, num_numbers):
    extracted_digits = []
    for i in range(num_numbers):
        extracted_digits.append(digits.data[offset + 10 * i])
        
    return extracted_digits

def extract_every_10th_target(offset, num_numbers):
    extracted_targets = []
    for i in range(num_numbers):
        extracted_targets.append(digits.target[offset + 10 * i])
        
    return extracted_targets

num_numbers = 3

digits_imgs = [extract_every_10th_digit(1, num_numbers),
               extract_every_10th_digit(8, num_numbers),
               extract_every_10th_digit(3, num_numbers),
               extract_every_10th_digit(9, num_numbers)
               ]

digits_targets = [extract_every_10th_target(1, num_numbers),
                  extract_every_10th_target(8, num_numbers),
                  extract_every_10th_target(3, num_numbers),
                  extract_every_10th_target(9, num_numbers)
               ]

fig, axes = plt.subplots(len(digits_imgs), 
                         len(digits_imgs[0]), 
                         figsize=(3*len(digits_imgs[0]), 3*len(digits_imgs)))

for i in range(len(digits_imgs)):
    for j in range(len(digits_imgs[0])):
        axes[i, j].imshow(digits_imgs[i][j].reshape(8,8), cmap='gray_r')
        axes[i, j].set_title("Target: " + str(digits_targets[i][j]))
#        plt.title(f'target:{digits.target[i]}')

plt.title("2D) Illustration of 8s, 1s, 9s, and 3s")
plt.show()

"""

It is unknown exactly what features the perceptron is pulling from this dataset, 
but from visual inspection:
1's and 8's don't look too different
Nor do 3's and 8's
Nor do 1's and 9's
9's and 8's look a bit different on the bottom-left most curve of the 8
"""





















