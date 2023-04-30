import numpy as np
import torch
import math
import csv
import json

from matplotlib import pyplot as plt

# FOR MNIST
# FOR LENET
# INITIAL CODE

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

def layers_to_unweighted_adj_matrix(blocks, block_sizes, total_size, mask=None):
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
    s = np.linalg.svd(W, full_matrices=False, compute_uv = False)

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
                
    return leverages

def regularize_A(A, r):
    sums = np.sum(A, axis=1)
    factors = np.zeros(len(sums))

    for i in range(len(factors)):
        if sums[i] != 0:
            factors[i] = r / sums[i]
    
    for i in range(len(A)):
        for j in range(len(A[0])):
            A[i][j] = A[i][j] * factors[i]
    
   #  print(np.sum(A, axis=1))
    return A

def get_mask_from_level(i):
    return "/replicate_1/level_" + str(i) + "/main/mask.pth"

def get_sparsity_report(i):
    return "/replicate_1/level_" + str(i) + "/main/sparsity_report.json"

def remove_disconnections(A):
    del_A = A

    zeros = []
    sums = np.sum(A, axis=1)

    for i in range(len(sums)):
        if sums[i] == 0:
            zeros.append(i)
    
    for i in range(len(zeros) - 1, -1, -1):
        del_A = np.delete(del_A, i, 0)
        del_A = np.delete(del_A, i, 1)

    return del_A

def get_fiedler(A):
    L = get_laplacian_matrix(A)

    w, v = np.linalg.eig(L)
    w.sort()
    return w[1]


PATH_START = "../open_lth_data/"
LEVELS = 20

PATHS = [
    "lottery_49d62984dbfd626736d5fe53513edac9",
    # "lottery_0fb9604a3c0ead8f41984073b4129f13"
]
DESCRIPTIONS = [
    # "mnist_lenet_100_50",
    "mnist_lenet_50_30"
]
PATH_END_LOTTO = "/replicate_1/level_0/main/model_ep0_it0.pth"
D = 5
with open('spectral_expansions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Model', 'Level', 'Sparsity', 'Spectral Expansion', 'Performance', 'is_lotto'])
    for index, p in enumerate(PATHS):
        PATH_LOTTO = PATH_START + p + PATH_END_LOTTO
        network = torch.load(PATH_LOTTO)

        for l in range(LEVELS):
            
            NUM_OUTPUTS = 10

            TOTAL_SIZE = 10
            PATH_MASK = PATH_START + p + get_mask_from_level(l)

            j = open(PATH_START + p + get_sparsity_report(l))

            sparsity_report = json.load(j)
            sparsity = sparsity_report["unpruned"] / sparsity_report["total"]
            mask = torch.load(PATH_MASK)

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

            A = layers_to_unweighted_adj_matrix(blocks, block_sizes, TOTAL_SIZE, mask=mask_blocks)
            # A = regularize_A(A, D)
            del_A = remove_disconnections(A)

            # print(np.shape(del_A))

            spectral_expansion = spectral_expansion_from_A(del_A)

            
            writer.writerow([DESCRIPTIONS[index], l, sparsity, spectral_expansion])


## THIS PART IS FOR LEVERAGE SCORES

with open('leverages.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Model', 'Level', 'Sparsity', 'Average Leverage Score', 'Leverage STDev', 'Performance', 'is_lotto'])
    for index, p in enumerate(PATHS):
        PATH_LOTTO = PATH_START + p + PATH_END_LOTTO
        network = torch.load(PATH_LOTTO)

        block_sizes = []
        blocks = []
            
        NUM_OUTPUTS = 10

        TOTAL_SIZE = 10

        for i in network:
            if 'weight' in i:
                TOTAL_SIZE += np.shape(network[i])[1]
                block_sizes.append(np.shape(network[i])[1])
                blocks.append(network[i])

        block_sizes.append(10)

        unmasked_A = layers_to_adj_matrix(blocks, block_sizes, TOTAL_SIZE)
        unmasked_leverages = leverage_from_A(unmasked_A)

        for l in range(LEVELS):

            PATH_MASK = PATH_START + p + get_mask_from_level(l)

            j = open(PATH_START + p + get_sparsity_report(l))

            sparsity_report = json.load(j)
            sparsity = sparsity_report["unpruned"] / sparsity_report["total"]
            mask = torch.load(PATH_MASK)

            mask_blocks = []

            for i in mask:
                if 'weight' in i:
                    mask_blocks.append(mask[i])

            masked_A = layers_to_unweighted_adj_matrix(blocks, block_sizes, TOTAL_SIZE, mask=mask_blocks)
            leverages = np.multiply(unmasked_leverages, masked_A)

            leverages = leverages.flatten()

            leverages = leverages[leverages != 0]

            # Creating histogram
            fig, ax = plt.subplots(figsize =(10, 7))
            ax.hist(leverages)
            # Adding extra features   
            plt.xlabel("Leverage Scores")
            plt.ylabel("Count")
            
            plt.title('Leverage Scores for ' + DESCRIPTIONS[index] + " At Sparsity of " + str(sparsity))
            # Show plot
            plt.savefig(DESCRIPTIONS[index] + "_" + str(l) + ".png")
            plt.show()

            writer.writerow([DESCRIPTIONS[index], l, sparsity, np.mean(leverages), np.std(leverages)])
            