# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 12:29:03 2023

@author: Keena
"""

from IPython import get_ipython


get_ipython().magic('clear')
get_ipython().magic('reset -f')
import numpy as np

# Create your original numpy.ndarray (replace this with your actual data)
original_array = np.array([[2, 3], [4, 5], [6, 7]])

# Determine the number of rows in the original array
num_rows = original_array.shape[0]

# Create a new column filled with 1's
ones_column = np.ones((num_rows, 1), dtype=int)

# Stack the original array and the ones_column horizontally
new_array = np.hstack((original_array, ones_column))

print(new_array)
