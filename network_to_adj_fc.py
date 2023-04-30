import numpy as np
import torch
import math

# FOR MNIST
# FOR LENET
# INITIAL CODE

PATH = '../open_lth_data/train_574e51abc295d8da78175b320504f2ba/replicate_1/main/model_ep0_it0.pth'

PATH_LOTTO = "../open_lth_data/lottery_184ace1d6901ace6854b0a595cbd6b27/replicate_1/level_3/main/model_ep0_it0.pth"
PATH_MASK = "../open_lth_data/lottery_184ace1d6901ace6854b0a595cbd6b27/replicate_1/level_3/main/mask.pth"

network = torch.load(PATH_LOTTO)
mask = torch.load(PATH_MASK)

NUM_OUTPUTS = 10

TOTAL_SIZE = 10
block_sizes = []
blocks = []

mask_blocks = []

for i in network:
    if 'weight' in i:
        TOTAL_SIZE += np.shape(network[i])[1]
        block_sizes.append(np.shape(network[i])[1])
        blocks.append(network[i])

for i in mask:
    if 'weight' in i:
        mask_blocks.append(mask[i])

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

def prune_matrix(a, sparsity = 0):
    for i in range(len(a)):
        for j in range(len(a[0])):
            if np.random.random(1)[0] < sparsity:
                a[i][j] = 0
    
    return a

def layers_to_adj_matrix(blocks, block_sizes, total_size, mask=None):
    adj = np.zeros((total_size, total_size))

    if not mask:
        mask = np.ones((len(blocks), total_size, total_size))
    
    curr_block = 0
    next_block = 0

    for b, block in enumerate(blocks):
        next_block += block_sizes[b]

        for i in range(len(block)):
            for j in range(len(block[0])):
                # TO the i'th neuron in the layer from the j'th in the prev layer
                adj[curr_block + j][next_block + i] = abs(block[i][j] * mask[b][i][j])
                adj[next_block + i][curr_block + j] = abs(block[i][j] * mask[b][i][j])

        curr_block += block_sizes[b]
    
    return adj

def layers_to__unweighted_adj_matrix(blocks, block_sizes, total_size, mask=None):
    adj = np.zeros((total_size, total_size))

    if not mask:
        mask = np.ones((len(blocks), total_size, total_size))
    
    curr_block = 0
    next_block = 0

    for b, block in enumerate(blocks):
        next_block += block_sizes[b]

        for i in range(len(block)):
            for j in range(len(block[0])):
                # TO the i'th neuron in the layer from the j'th in the prev layer
                if mask[b][i][j] != 0 and block[i][j] != 0:
                    adj[curr_block + j][next_block + i] = 1
                    adj[next_block + i][curr_block + j] = 1

        curr_block += block_sizes[b]
    
    return adj

def spectral_expansion_from_A(A):
    if len(A) < 2:
        return -1

    W = get_random_walk_matrix(A)
    u, s, vh = np.linalg.svd(W, full_matrices=True)

    return 1 - s[1]

def leverage_from_A(A):
    L = get_laplacian_matrix(A)

    L_inv = np.linalg.pinv(L)

    leverages = np.zeros(np.shape(A))
    counter = 0
    for a in range(len(leverages)):
        for b in range(len(leverages[0])):
            if A[a][b] != 0:
                delta_a = np.matrix(np.zeros(len(A))).T
                delta_b = np.matrix(np.zeros(len(A))).T

                delta_a[a] = 1
                delta_b[b] = 1

                # print(np.shape(L_inv))
                # print(np.shape(delta_a - delta_b))
                temp = L_inv @ (delta_a - delta_b)
                # Effective res
                ER = (delta_a - delta_b).T @ temp 
                # Leverage
                leverages[a][b] = A[a][b] * ER
                counter += 1
                
    print(counter)
    return leverages

A = layers_to_adj_matrix(blocks, block_sizes, TOTAL_SIZE, mask=mask_blocks)

leverages = leverage_from_A(A)
print(leverages)
print(np.sum(leverages, axis=1))

# print(np.sum(A))

# print(np.sum(A, axis=1))

# W = get_random_walk_matrix(A)

# print(W)
# print(np.sum(W, axis=1))
# print(np.sum(W))

# u, s, vh = np.linalg.svd(W, full_matrices=True)
# print(s)
# unpruned_A = layers_to_adj_matrix(blocks, block_sizes, TOTAL_SIZE)
# pruned_A = prune_matrix(A, 0.5)

# W = get_random_walk_matrix(pruned_A)

# u, s, vh = np.linalg.svd(W, full_matrices=True)
# print(s)

