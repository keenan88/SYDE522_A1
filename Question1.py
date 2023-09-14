# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:18:58 2023

@author: Keena
"""

import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython


get_ipython().magic('clear')
get_ipython().magic('reset -f')


def compare_results(desired_output, actual_results):
    accurate_classifications = 0

    for index in range(len(desired_output)):
        if desired_output[index] ==actual_results[index]:
            accurate_classifications += 1
            
    pct_accuracy = 100 * accurate_classifications / len(desired_output)
    
    return pct_accuracy
    
def plot_results(input_data, classifications, weights):
    plt.figure(figsize=(6,6))
    plt.scatter(input_data[:,0], input_data[:,1], c=np.where(classifications, class_0_clr, class_1_clr))
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
    if weights is not -1:
        boundary_x  = np.linspace(-3, 5, 100)
        boundary_y = -(weights[0] * boundary_x + weights[2]) / weights[1]
             
        plt.plot(boundary_x, boundary_y, label='Line', color='grey')
    
    plt.xlim(-3, 5)
    plt.ylim(-3, 6)
    plt.show()
    
def classify_points(points, weights):
    classifications = []
    
    for idx, point in enumerate(points):
        
        score = np.dot(weights, point)
        classification = class_0 if score < 0 else class_1
        classifications.append(classification)
        
    return classifications
    
def full_learning_iteration(input_data, desired_output, weights, learning_rate):
    
    for idx, point in enumerate(input_data):
        
        score = np.dot(weights, point)
        classification = class_0 if score < 0 else class_1
        
        if classification != desired_output[idx]:
            delta_w = learning_rate * point * (desired_output[idx] - classification)
        
            weights += delta_w   
            
    return weights
    

red_class_num = 0
blue_class_num = 1

class_0 = 0
class_0_clr = 'blue'

class_1 = 1
class_1_clr = 'red'

input_data, desired_output = sklearn.datasets.make_blobs(centers=[[-2, -2], [2, 2]], 
                                             cluster_std=[0.3, 1.5], 
                                             random_state=0, 
                                             n_samples=200, 
                                             n_features=2)

theta_column = np.ones((input_data.shape[0], 1), dtype=int)
input_data = np.hstack((input_data, theta_column))


plot_results(input_data, desired_output, -1)


weights= [1, -1, 0]
learning_rate = 0.1

# A)
print("A) ")

model_output_no_training = classify_points(input_data, weights)
    
plot_results(input_data, model_output_no_training, weights)

compare_results(desired_output,  model_output_no_training)
    
print("Weights: ", weights)
print()

# B)
print("B) ")


weights = full_learning_iteration(input_data, desired_output, weights, learning_rate)
    
model_output_trained = classify_points(input_data, weights)

pct_accuracy = compare_results(desired_output, model_output_trained)
print("Weights: ", weights)
print("Accuracy: ", pct_accuracy, "%")
print()
plot_results(input_data, model_output_trained, weights)

# C)
print("C) ")

weights = [1, -1, 0]
pct_accuracy = 0
repetitions = 0

while pct_accuracy < 100:
    
    weights = full_learning_iteration(input_data, desired_output, weights, learning_rate)
            
    model_output_trained = classify_points(input_data, weights)

    pct_accuracy = compare_results(desired_output, model_output_trained)
    print(pct_accuracy)
    
    repetitions += 1

print("Repetitions: ", repetitions)
print("Weights: ", weights)
print("Accuracy: ", pct_accuracy, "%")
print()
plot_results(input_data, model_output_trained, weights)

# D)
print("D) ")

learning_rates = [1, 0.01, 100]

for learning_rate in learning_rates:
    repetitions = 0
    weights = [1, -1, 0]
    pct_accuracy = 0
    
    while pct_accuracy < 100:
        
        weights = full_learning_iteration(input_data, desired_output, weights, learning_rate)
                
        model_output_trained = classify_points(input_data, weights)
    
        pct_accuracy = compare_results(desired_output, model_output_trained)
                
        repetitions += 1
    
    print("Learning Rate: ", learning_rate)
    print("Repetitions: ", repetitions)
    print("Weights: ", weights)
    print("Accuracy: ", pct_accuracy, "%")
    print()
    plot_results(input_data, model_output_trained, weights)


    



 
 
 
 
 
 
 

