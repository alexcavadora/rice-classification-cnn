import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS")
    else:
        device = torch.device('cpu')
        print("CPU")
    return device

