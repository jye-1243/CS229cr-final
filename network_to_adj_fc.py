import numpy as np
import torch
import math

# FOR MNIST
# FOR LENET
# INITIAL CODE

PATH = '../open_lth_data/train_574e51abc295d8da78175b320504f2ba/replicate_1/main/model_ep40_it0.pth'

PATH_LOTTO = "../open_lth_data/lottery_93bc65d66dfa64ffaf2a0ab105433a2c/replicate_1/level_0/main/model_ep0_it0.pth"
PATH_MASK = "../open_lth_data/lottery_93bc65d66dfa64ffaf2a0ab105433a2c/replicate_1/level_0/main/mask.pth"

network = torch.load(PATH)

NUM_OUTPUTS = 10

TOTAL_SIZE = 10
block_sizes = []
blocks = []

for i in network:
    if 'weight' in i:
        TOTAL_SIZE += np.shape(network[i])[1]
        block_sizes.append(np.shape(network[i])[1])
        blocks.append(network[i])

block_sizes.append(10)


def get_degree_matrix(a):
    degrees = np.sum(a, axis=1)

    D = np.zeros(np.shape(a))

    for i in range(np.shape(a)[0]):
        D[i][i] = degrees[i]
    
    return D

def get_laplacian_matrix(a):
    return get_degree_matrix(a) - a

def get_random_walk_matrix(a):
    W = np.zeros(np.shape(a))
    D = get_degree_matrix(a)
    for i in range(np.shape(D)[0]):
        for j in range(np.shape(D)[1]):
            if a[i][j] != 0:
                W[i][j] = a[i][j] / D[i][i]

    return W

def layers_to_adj_matrix(blocks, block_sizes, total_size, mask=None):
    adj = np.zeros((total_size, total_size))

    if not mask:
        mask = np.ones((total_size, total_size))
    
    curr_block = 0
    next_block = 0

    for b, block in enumerate(blocks):
        next_block += block_sizes[b]
        if b == 0:
            next_block -= 1

        print(curr_block)
        # print(len(block))
        print(next_block)
        # print(len(block[0]))



        for i in range(len(block)):
            for j in range(len(block[0])):
                # TO the i'th neuron in the layer from the j'th in the prev layer
                adj[curr_block + j][next_block + i] = abs(block[i][j] * mask[i][j])
                adj[next_block + i][curr_block + j] = abs(block[i][j] * mask[i][j])

        curr_block += block_sizes[b]
    
    return adj

A = layers_to_adj_matrix(blocks, block_sizes, TOTAL_SIZE)

print(np.sum(A, axis=1))

W = get_random_walk_matrix(A)

print(W)
print(np.sum(W, axis=1))
print(np.sum(W))

u, s, vh = np.linalg.svd(W, full_matrices=True)

print(s)
