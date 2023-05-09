import numpy as np
import torch
import math
import csv
import json

from matplotlib import pyplot as plt

THRESHOLD = 17
'''
CODE FOR PLOTTING: Sparsity vs. Fiedler
'''
  
x = []
y = []
x_rand = []
y_rand = []
  
with open('fiedler_3.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
      
    for row in plots:
        x.append(row[2])
        y.append(row[3])

for i,v in enumerate(x):
    if v == '0j':
        x[i] = 0
    elif v[0] == '(':
        x[i] = v[1:-4]

x = [float(i) for i in x[1:]]

for i,v in enumerate(y):
    if v == '0j':
        y[i] = 0
    elif v[0] == '(':
        y[i] = v[1:-4]

y = [float(i) for i in y[1:]]

with open('fiedler_3_randomized.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
      
    for row in plots:
        x_rand.append(row[3])
        y_rand.append(row[4])


for i,v in enumerate(x_rand):
    if v == '0j':
        x_rand[i] = 0
    elif v[0] == '(':
        x_rand[i] = v[1:-4]

for i,v in enumerate(y_rand):
    if v == '0j':
        y_rand[i] = 0
    elif v[0] == '(':
        y_rand[i] = v[1:-4]


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
plt.ylabel('Fiedler Value')
plt.title('Sparsity vs. Fiedler Value, mnist_lenet_100')
plt.legend()
plt.savefig('mnist_lenet_100_fiedler_revised.png')
plt.show()