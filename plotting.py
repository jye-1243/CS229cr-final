import numpy as np
import torch
import math
import csv
import json

from matplotlib import pyplot as plt

THRESHOLD = 17
'''
CODE FOR PLOTTING: Sparsity vs. Spectral Expansion
'''
  
x = []
y = []
x_rand = []
y_rand = []
  
with open('spectral_expansions_2.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
      
    for row in plots:
        x.append(row[2])
        y.append(row[3])

x = [float(i) for i in x[1:]]
y = [float(i) for i in y[1:]]

with open('spectral_expansions_randomized_2.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
      
    for row in plots:
        x_rand.append(row[3])
        y_rand.append(row[4])

x_rand = np.array([float(i) for i in x_rand[1:]])
x_rand = np.average(x_rand.reshape(-1, 5), axis=1)
print(x_rand)

y_rand = np.array([float(i) for i in y_rand[1:]])
y_rand_std = np.std(y_rand.reshape(-1, 5), axis=1)
y_rand = np.average(y_rand.reshape(-1, 5), axis=1)
print(y_rand)
print(y_rand_std)

plt.scatter(x, y, c='orange', marker='x', label="lottery")
plt.errorbar(x_rand, y_rand, yerr=(2*y_rand_std), linestyle=None, c='blue', marker = "o", label = "random submatrix")
plt.xlabel('Sparsity')
plt.ylabel('Spectral Expansion')
plt.title('Sparsity vs. Spectral Expansion, mnist_lenet_100_50')
plt.legend()
plt.savefig('mnist_lenet_100_50_sparsity_spectral_revised.png')
plt.show()