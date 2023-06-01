import torch
from train import  train


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("successful")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("gpu unrun")
    train()
# print(torch.cuda.is_available())