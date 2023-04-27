import torch

PATH = '../open_lth_data/train_71bc92a970b64a76d7ab7681764b0021/replicate_1/main/model_ep160_it0.pth'

checkpoint = torch.load(PATH)

print(len(checkpoint))

for i in checkpoint:
    print(i)


print(checkpoint['conv.weight'])