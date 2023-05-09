import numpy as np
import torch
import math
import csv
import json

from matplotlib import pyplot as plt

THRESHOLD = 17


PERFORMANCE_1 = [0.9729, 0.9725, 0.9729, 0.9736, 0.9728, 0.9733, 0.9723, 0.9717, 0.9725, 0.9706, 0.9695, 0.97, 0.9707, 0.9687, 0.9649, 0.963, 0.957, 0.9494, 0.9411, 0.9183, 0.8504]
PERFORMANCE_2 = [0.9776, 0.9797, 0.98, 0.9798, 0.979, 0.9788, 0.9781, 0.9771, 0.9784, 0.9777, 0.9783, 0.9775, 0.9778, 0.9766, 0.9769, 0.9739, 0.9701, 0.9685, 0.9652, 0.9597, 0.9463]
PERFORMANCE_3 = [0.979, 0.9793, 0.9793, 0.9788, 0.9789, 0.9789, 0.9792, 0.9796, 0.9792, 0.9793, 0.9804, 0.9804, 0.9787, 0.9782, 0.976, 0.9756, 0.9741, 0.9689, 0.9631, 0.9126]

x = []
y = []
x_rand = []
y_rand = []
  
with open('spectral_expansions_2.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
      
    for row in plots:
        x.append(row[2])
        y.append(row[3])

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

# with open('fiedler_3_randomized.csv','r') as csvfile:
#     plots = csv.reader(csvfile, delimiter = ',')
      
#     for row in plots:
#         x_rand.append(row[3])
#         y_rand.append(row[4])


# for i,v in enumerate(x_rand):
#     if v == '0j':
#         x_rand[i] = 0
#     elif v[0] == '(':
#         x_rand[i] = v[1:-4]

# for i,v in enumerate(y_rand):
#     if v == '0j':
#         y_rand[i] = 0
#     elif v[0] == '(':
#         y_rand[i] = v[1:-4]


# x_rand = np.array([float(i) for i in x_rand[1:]])
# x_rand = np.average(x_rand.reshape(-1, 5), axis=1)
# print(x_rand)

# y_rand = np.array([float(i) for i in y_rand[1:]])
# y_rand_std = np.std(y_rand.reshape(-1, 5), axis=1)
# y_rand = np.average(y_rand.reshape(-1, 5), axis=1)
# print(y_rand)
# print(y_rand_std)

print(y)
fig, ax1 = plt.subplots()
ax1.set_xlabel("Sparsity (Log Scale)'")
ax1.set_title("Test Accuracy and Spectral Expansion Delta, mnist_lenet_100_50")

ax2 = ax1.twinx() 

ax1.plot(x, np.absolute(y - y_rand), c='blue', marker='o', label="Spectral Expansion Delta")
ax2.plot(x, np.array(PERFORMANCE_2), c='orange', marker='x', label="Performance")
ax1.set_ylabel("Spectral Expansion Delta Value")
ax2.set_ylabel('Performance on Test Data')
plt.xscale("log")
ax1.legend(loc=4)
ax2.legend(loc=0)
plt.savefig('revised_mnist_lenet_100_50_performance_v_spectral.png')
plt.show()