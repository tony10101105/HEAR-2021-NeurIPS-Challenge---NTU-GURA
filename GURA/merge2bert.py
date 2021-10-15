import torch
from transformers import Wav2Vec2Model, HubertModel

class hubert_wav2vec2_avg(torch.nn.Module):
    def __init__(self):
        super(hubert_wav2vec2, self).__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ll60k")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        
    def forward(self, x):
        out1 = self.hubert(x)
        out2 = self.wav2vec2(x)
        out = (out1.last_hidden_state + out2.last_hidden_state) / 2
        return out

class hubert_wav2vec2_cat(torch.nn.Module):
    def __init__(self):
        super(hubert_wav2vec2_cat, self).__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ll60k")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        
    def forward(self, x):
        out1 = self.hubert(x)
        out2 = self.wav2vec2(x)
        out = torch.cat((out1.last_hidden_state, out2.last_hidden_state), dim=2)
        return out
