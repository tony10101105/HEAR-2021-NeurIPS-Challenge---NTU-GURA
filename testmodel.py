import torch
from hubert_ft import load_model

a = load_model()
torch.save(a.state_dict(), 'toyzdog.pt')
'''
b = load_model(model_hub="facebook/wav2vec2-large-xlsr-53")
torch.save(b.state_dict(), '/pretrained/weedtoyz.pt')
'''
