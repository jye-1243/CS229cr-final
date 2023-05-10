import numpy as np
import torch
import math
import csv
import json

from matplotlib import pyplot as plt

# FOR MNIST
# FOR LENET
# INITIAL CODE

'''Functions for Matrices'''

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

def get_regularized_matrix_for_expansion(a):
    D = get_degree_matrix(a)
    D_half = np.zeros(np.shape(D))

    for i in range(len(D)):
        if D[i][i] != 0:
            D_half[i][i] = 1/np.sqrt(D[i][i])
        
    W = D_half @ a @ D_half
    # for i in range(len(W)):
    #     for j in range(len(W[0])):
    #         if abs(W[i][j] - W[j][i]) > 0.001:
    #             print("BAD")

    return W

def prune_matrix(a, sparsity = 1):
    for i in range(len(a)):
        for j in range(i+1, len(a[0])):
            if np.random.random(1)[0] < (1 - sparsity):
                a[i][j] = 0
                a[j][i] = 0
    
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

    W = get_regularized_matrix_for_expansion(A)

    degrees1 = np.sum(W, axis=0)
    degrees2 = np.sum(W, axis=1)

    s = np.linalg.svd(W, full_matrices=True, compute_uv = False)
    # print(s)

    return 1 - s[2]

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
        del_A = np.delete(del_A, zeros[i], 0)
        del_A = np.delete(del_A, zeros[i], 1)
        
    return del_A

def remove_disconnections_tree(A):
    del_A = []

    visited = []

    for i in range(1, 11):
        if (len(A) - i) not in visited:
            q = [len(A)-i]
            visited.append(len(A) - i)

            while len(q) > 0:
                curr = q.pop(0)

                for j in range(len(A)):
                    if A[curr][j] != 0 and j not in visited:
                        q.append(j)
                        visited.append(j)

    visited = sorted(visited)

    for i in range(len(visited)):
        del_A.append(A[i])
    
    for i in range(len(del_A)):
        del_A[i] = del_A[i][visited]

    return np.array(del_A)

def get_fiedler(A):
    L = get_laplacian_matrix(A)

    w, v = np.linalg.eig(L)
    w.sort()
    return w[1]


'''Lottery Ticket Loading'''

PATH_START = "../open_lth_data/"
LEVELS = 20

PATHS = [
    "lottery_49d62984dbfd626736d5fe53513edac9",
    #"lottery_0fb9604a3c0ead8f41984073b4129f13"
    #"lottery_dd39712abe0934c13324a77320fe238c"
]
DESCRIPTIONS = [
    "mnist_lenet_50_30"
    #"mnist_lenet_100_50",
    #"mnist_lenet_100"
]
PATH_END_LOTTO = "/replicate_1/level_0/main/model_ep0_it0.pth"

'''FIEDLER '''
# # THIS IS FOR FIEDLER

# with open('fiedler_3.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow(['Model', 'Level', 'Sparsity', 'Fiedler'])
#     for index, p in enumerate(PATHS):
#         PATH_LOTTO = PATH_START + p + PATH_END_LOTTO
#         network = torch.load(PATH_LOTTO)

#         NUM_OUTPUTS = 10

#         TOTAL_SIZE = 10
#         block_sizes = []
#         blocks = []

#         for i in network:
#             if 'weight' in i:
#                 TOTAL_SIZE += np.shape(network[i])[1]
#                 block_sizes.append(np.shape(network[i])[1])
#                 blocks.append(network[i])

#         block_sizes.append(10)
#         for l in range(LEVELS):
#             print(l)
            
#             PATH_MASK = PATH_START + p + get_mask_from_level(l)
#             j = open(PATH_START + p + get_sparsity_report(l))
#             sparsity_report = json.load(j)
#             sparsity = sparsity_report["unpruned"] / sparsity_report["total"]
#             mask = torch.load(PATH_MASK)

#             mask_blocks = []

#             for i in mask:
#                 if 'weight' in i:
#                     mask_blocks.append(mask[i])

#             A = layers_to_adj_matrix(blocks, block_sizes, TOTAL_SIZE, mask=mask_blocks)
#             del_A = remove_disconnections_tree(A)

#             fiedler = get_fiedler(del_A)
#             writer.writerow([DESCRIPTIONS[index], l, sparsity, fiedler])

# with open('fiedler_3_randomized.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow(['Model', 'Level', "Exp No", 'Sparsity', 'Fiedler'])
#     for index, p in enumerate(PATHS):
#         PATH_LOTTO = PATH_START + p + PATH_END_LOTTO
#         network = torch.load(PATH_LOTTO)

#         NUM_OUTPUTS = 10

#         TOTAL_SIZE = 10
#         block_sizes = []
#         blocks = []

#         for i in network:
#             if 'weight' in i:
#                 TOTAL_SIZE += np.shape(network[i])[1]
#                 block_sizes.append(np.shape(network[i])[1])
#                 blocks.append(network[i])

#         block_sizes.append(10)
#         for l in range(LEVELS):
#             print(l)
            
#             PATH_MASK = PATH_START + p + get_mask_from_level(l)
#             j = open(PATH_START + p + get_sparsity_report(l))
#             sparsity_report = json.load(j)
#             sparsity = sparsity_report["unpruned"] / sparsity_report["total"]
#             mask = torch.load(PATH_MASK)

#             mask_blocks = []

#             for i in mask:
#                 if 'weight' in i:
#                     mask_blocks.append(mask[i])
            
#             for n in range(5):
#                 A = layers_to_adj_matrix(blocks, block_sizes, TOTAL_SIZE, mask=mask_blocks)
#                 A = prune_matrix(A, sparsity)
#                 del_A = remove_disconnections_tree(A)

#                 fiedler = get_fiedler(del_A)
#                 writer.writerow([DESCRIPTIONS[index], l, n, sparsity, fiedler])

# # _______________________________________________________________________________________________________
# '''SPECTRAL '''
# with open('spectral_expansions_randomized_1.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     writer.writerow(['Model', 'Level',  'Exp No', 'Sparsity', 'Spectral Expansion', 'Performance', 'is_lotto'])
#     for index, p in enumerate(PATHS):
#         PATH_LOTTO = PATH_START + p + PATH_END_LOTTO
#         network = torch.load(PATH_LOTTO)

#         for l in range(LEVELS):
#             print(l)
            
#             NUM_OUTPUTS = 10

#             TOTAL_SIZE = 10
#             PATH_MASK = PATH_START + p + get_mask_from_level(l)

#             j = open(PATH_START + p + get_sparsity_report(l))

#             sparsity_report = json.load(j)
#             sparsity = sparsity_report["unpruned"] / sparsity_report["total"]
#             mask = torch.load(PATH_MASK)

#             block_sizes = []
#             blocks = []

#             mask_blocks = []

#             for i in network:
#                 if 'weight' in i:
#                     TOTAL_SIZE += np.shape(network[i])[1]
#                     block_sizes.append(np.shape(network[i])[1])
#                     blocks.append(network[i])

#             # for i in mask:
#             #     if 'weight' in i:
#             #         mask_blocks.append(mask[i])

#             block_sizes.append(10)

#             for n in range(5):
#                 A = layers_to_unweighted_adj_matrix(blocks, block_sizes, TOTAL_SIZE)
#                 A = prune_matrix(A, sparsity)
#                 del_A = remove_disconnections(A)

#                 expansion = spectral_expansion_from_A(del_A)
#                 writer.writerow([DESCRIPTIONS[index], l, n, sparsity, expansion])

            
#             # A = layers_to_unweighted_adj_matrix(blocks, block_sizes, TOTAL_SIZE, mask=mask_blocks)
#             # # A = prune_matrix(A, sparsity)
#             # del_A = remove_disconnections(A)

#             # spectral_expansion = spectral_expansion_from_A(del_A)
            
#             # writer.writerow([DESCRIPTIONS[index], l, sparsity, spectral_expansion])

# '''Other Stuff'''

#             # A = layers_to_unweighted_adj_matrix(blocks, block_sizes, TOTAL_SIZE, mask=mask_blocks)
#             # # A = regularize_A(A, D)
#             # del_A = remove_disconnections(A)

#             # # print(np.shape(del_A))

#             # spectral_expansion = spectral_expansion_from_A(del_A)

            
#             # writer.writerow([DESCRIPTIONS[index], l, sparsity, spectral_expansion])


# ## THIS PART IS FOR LEVERAGE SCORES

with open('leverages_1.csv', 'w', newline='') as csvfile:
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
            ax.hist(leverages, density=True)
            # Adding extra features   
            plt.xlabel("Leverage Scores")
            plt.ylabel("Density")
            
            plt.title('Leverage Scores for ' + DESCRIPTIONS[index] + " At Sparsity of " + str(sparsity))
            # Show plot
            plt.savefig("revised_" + DESCRIPTIONS[index] + "_" + str(l) + ".png")
            plt.show()

            writer.writerow([DESCRIPTIONS[index], l, sparsity, np.mean(leverages), np.std(leverages)])


# print(spectral_expansion_from_A(np.array([[1, 7, 4], [7, 6, 9], [4, 9, 1]])))