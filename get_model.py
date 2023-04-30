import numpy as np
import torch

PATH = '../open_lth_data/train_574e51abc295d8da78175b320504f2ba/replicate_1/main/model_ep40_it0.pth'

PATH_LOTTO = "../open_lth_data/lottery_93bc65d66dfa64ffaf2a0ab105433a2c/replicate_1/level_0/main/model_ep0_it0.pth"
PATH_MASK = "../open_lth_data/lottery_93bc65d66dfa64ffaf2a0ab105433a2c/replicate_1/level_3/main/mask.pth"

checkpoint = torch.load(PATH)

print(len(checkpoint))

for i in checkpoint:
    print(i)
    print(np.shape(checkpoint[i]))


print(checkpoint['blocks.0.conv1.weight'])
# print(np.shape(checkpoint['conv.weight']))
