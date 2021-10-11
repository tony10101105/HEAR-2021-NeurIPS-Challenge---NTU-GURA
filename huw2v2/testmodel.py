import torch
from huw2v2 import load_model

a = load_model()
torch.save(a.state_dict(), 'huw2v2.pt')
