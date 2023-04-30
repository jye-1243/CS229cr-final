import numpy as np
import torch

# FOR CIFAR 10
# FOR RESNET 20
# INITIAL CODE

PATH = '../open_lth_data/train_1/replicate_1/main/model_ep160_it0.pth'

PATH_LOTTO = "../open_lth_data/lottery_93bc65d66dfa64ffaf2a0ab105433a2c/replicate_1/level_0/main/model_ep0_it0.pth"
PATH_MASK = "../open_lth_data/lottery_93bc65d66dfa64ffaf2a0ab105433a2c/replicate_1/level_0/main/mask.pth"

network = torch.load(PATH_LOTTO)

# Input size
INPUT_SIZE = 32 * 32 * 3

# After conv
# First sequence size
SEQ1_SIZE = 32 * 32 * 16

# AFTER first set of blocks
SEQ2_SIZE = 16 * 16 * 32

# AFTER second set of blocks
SEQ3_SIZE = 8 * 8 * 64

# Flattened
FLAT_SIZE = 64

# Final layer
OUTPUT_SIZE = 10

NUM_PER_SEQ = 3

# Then get total number of vertices
ARR_SIZE = INPUT_SIZE + NUM_PER_SEQ * (SEQ1_SIZE + SEQ2_SIZE + SEQ3_SIZE) + FLAT_SIZE + OUTPUT_SIZE

# Kinda ignoring batch layers here since they aren't masked anyways
def conv_weights_to_adj_with_padding(w, size1, size2, channels1, channels2, k = 3, stride=1, mask=None):
    # Sizes are image sizes
    # Channels = depth
    # k is kernel size
    # Stride is stride lol

    if not mask:
        mask = np.ones(np.shape(w))

    im1 = size1 * size1
    im2 = size2 * size2

    # Nodes per layer
    n1 = size1 * size1 * channels1
    n2 = size2 * size2 * channels2

    # Total number of nodes
    n = n1 + n2

    adj = np.zeros((n, n))

    # counter = 0

    # How we index into adj matrix?
    # I think we do it channel by channel
    # So the first size1*size1 values in n1 are for channel1, then channel2, etc
    # This is because we apply convolutions layer by layer
    # Then we index by row then by column

    # Weights are split by dimension [output dimensions, inputs, kernel, kernel]
    # AKA [channels2, channels1, k, k]

    for output_dim, output_weights in enumerate(w):
        for input_dim, input_weights_tensor in enumerate(output_weights):
            # Now input weights should be a k x k matrix which we can duplicate across a bunch of vertices
            # Now we want to iterate over inputs to map to the corresponding output
            input_weights = input_weights_tensor.cpu()

            for i in range(0, size1, stride):
                for j in range(0, size1, stride):
                    central_input = input_dim * im1 + size1 * i + j
                    output_vertex = n1 + output_dim * im2 + size2 * i + j

                    if k == 1:
                        adj[central_input][output_vertex] = input_weights[0][0] * mask[output_dim][input_dim][0][0]
                        adj[output_vertex][central_input] = input_weights[0][0] * mask[output_dim][input_dim][0][0]
                        # counter += 1
                    else:
                        adj[central_input][output_vertex] = input_weights[1][1] * mask[output_dim][input_dim][1][1]
                        adj[output_vertex][central_input] = input_weights[1][1] * mask[output_dim][input_dim][1][1]
                        # counter += 1

                        if i > 0:
                            adj[central_input - size1][output_vertex] = input_weights[0][1] * mask[output_dim][input_dim][0][1]
                            adj[output_vertex][central_input - size1] = input_weights[0][1] * mask[output_dim][input_dim][0][1]
                            # counter += 1

                            if j > 0:
                                adj[central_input - 1 - size1][output_vertex] = input_weights[0][0] * mask[output_dim][input_dim][0][0]
                                adj[output_vertex][central_input - 1 - size1] = input_weights[0][0] * mask[output_dim][input_dim][0][0]
                                # counter += 1
                            if j < size1 - 1:
                                adj[central_input + 1 - size1][output_vertex] = input_weights[0][2] * mask[output_dim][input_dim][0][2]
                                adj[output_vertex][central_input + 1 - size1] = input_weights[0][2] * mask[output_dim][input_dim][0][2]
                                # counter += 1
                        
                        if i < size1 - 1:
                            adj[central_input + size1][output_vertex] = input_weights[2][1] * mask[output_dim][input_dim][2][1]
                            adj[output_vertex][central_input + size1] = input_weights[2][1] * mask[output_dim][input_dim][2][1]
                            # counter += 1

                            if j > 0:
                                adj[central_input - 1 + size1][output_vertex] = input_weights[2][0] * mask[output_dim][input_dim][2][0]
                                adj[output_vertex][central_input - 1 + size1]= input_weights[2][0] * mask[output_dim][input_dim][2][0]
                                # counter += 1
                            if j < size1 - 1:
                                adj[central_input + 1 + size1][output_vertex] = input_weights[2][2] * mask[output_dim][input_dim][2][2]
                                adj[output_vertex][central_input + 1 + size1] = input_weights[2][2] * mask[output_dim][input_dim][2][2]
                                # counter += 1
                        
                        if j > 0:
                            adj[central_input - 1][output_vertex] = input_weights[1][0] * mask[output_dim][input_dim][1][0]
                            adj[output_vertex][central_input - 1] = input_weights[1][0] * mask[output_dim][input_dim][1][0]
                            # counter += 1
                        if j < size1 - 1:
                            adj[central_input + 1][output_vertex] = input_weights[1][2] * mask[output_dim][input_dim][1][0]
                            adj[output_vertex][central_input + 1] = input_weights[1][2] * mask[output_dim][input_dim][1][0]
                            # counter += 1
    # print(counter)
    return adj

a = conv_weights_to_adj_with_padding(network['conv.weight'], 32, 32, 3, 16)
print(np.shape(a))
print(a)

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

W = get_random_walk_matrix(a)
print(np.sum(W, axis=1))
print(W)

u, s, vh = np.linalg.svd(W, full_matrices=True)
print(s)

# print(np.sum(get_random_walk_matrix(a), axis=1))

# After first conv
# Input is 32x32x3
# Then the conv layer is S = 1, padding=1
# Should be 32 x 32 x 16 image?

# Then we do a batch norm layer
# This keeps shape I think

# Size change on block 3
# Size change on block 6
# This is with the downsample stuff

# After block 8, this is a length 64 output
# Fully connected layer
# To 10 values
