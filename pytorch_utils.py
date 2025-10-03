import torch

def get_device():
    dev_name = ""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        dev_name = "CUDA"
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        dev_name = "MPS"
    else:
        device = torch.device('cpu')
        dev_name = "CPU"
    print("Using device: ", dev_name)
    return device
