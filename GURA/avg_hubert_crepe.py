from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

####### Crepe #######
import torchcrepe
SAMPLE_RATE = 16000

TIMESTAMP_HOP_SIZE = 50
SCENE_HOP_SIZE = 250

TIMESTAMP_HOP_SIZE_SAMPLES = (SAMPLE_RATE * TIMESTAMP_HOP_SIZE) // 1000
SCENE_HOP_SIZE_SAMPLES = (SAMPLE_RATE * SCENE_HOP_SIZE) // 1000

####### Hubert #######
from transformers import HubertModel

class TorchCrepeModel(torch.nn.Module):
    """
    A pretty gross wrapper on torchcrepe, because of its implicit singleton
    model loading: https://github.com/maxrmorrison/torchcrepe/issues/13
    """

    # sample rate and embedding sizes are required model attributes for the HEAR API
    sample_rate = SAMPLE_RATE

    def __init__(self):
        super().__init__()

        # This is gross.
        if torch.cuda.is_available():
            torchcrepe.load.model(device="cuda", capacity="full")
        else:
            torchcrepe.load.model(device="cpu", capacity="full")

    def forward(self, x: Tensor, hop_size_samples: int):
        # Or do x.device?
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        if x.ndim == 1:
            x = x.view(1, x.shape[0])

        assert x.ndim == 2

        # This is lame, sorry
        # torchcrepe only can process one audio at a time
        embeddings = []
        for i in range(x.shape[0]):
            embedding = torchcrepe.embed(
                audio=x[i].view(1, x.shape[1]),
                sample_rate=self.sample_rate,
                hop_length=hop_size_samples,
                model="full",
                device=device,
                pad=True,
                # Otherwise dcase exceeds memory on a V100
                batch_size=512,
            )
            # Convert 1 x frames x 32x64 embedding to 1 x frames x 32*64
            assert embedding.shape[0] == 1
            assert embedding.ndim == 4
            embedding = embedding.view((1, embedding.shape[1], -1))
            embeddings.append(embedding)
        return torch.cat(embeddings)

class XC(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-xlarge-ll60k")
        self.crepe = TorchCrepeModel()

    def forward(self, x, hop_size_samples):
        hubert_output = self.hubert(x).last_hidden_state
        crepe_output = self.crepe(x, hop_size_samples)

        return hubert_output, crepe_output

def load_model(model_file_path: str = "") -> torch.nn.Module:
    """
    Args:
        model_file_path: Ignored
    Returns:
        XC()
    """

    model = XC()

    model.sample_rate = SAMPLE_RATE

    model.timestamp_embedding_size = 1024
    model.scene_embedding_size = 1024

    return model

def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
    hop_size_samples: int = TIMESTAMP_HOP_SIZE_SAMPLES,
) -> Tuple[Tensor, Tensor]:

    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (n_sounds, num_samples)"
        )

    if not isinstance(model, XC):
        raise ValueError(f"Model must be an instance of {XC.__name__}")

    model.eval()
    with torch.no_grad():
        xlarge_embeddings, crepe_embeddings = model(audio, hop_size_samples)


    """
    resample embeddings to same shape as xlarge_embeddings
    """
    xlarge_embeddings = F.interpolate(xlarge_embeddings,
                            size = model.scene_embedding_size,
                            mode = "linear"
                            )
    crepe_embeddings = F.interpolate(crepe_embeddings,
                            size = model.scene_embedding_size,
                            mode = "linear"
                            )
    crepe_embeddings = F.interpolate(crepe_embeddings.permute(0, 2, 1),
                            size = xlarge_embeddings.shape[1],
                            mode = "linear"
                        ).permute(0, 2, 1)

    audio_ms = int(audio.shape[1] / model.sample_rate * 1000)
    ntimestamps = (audio_ms - 5) // 20
    last_center = 12.5 + (ntimestamps - 1) * 20
    timestamps = torch.arange(12.5, last_center + 20, 20)
    assert len(timestamps) == ntimestamps
    timestamps = timestamps.expand((xlarge_embeddings.shape[0], timestamps.shape[0]))
    assert timestamps.shape[1] == xlarge_embeddings.shape[1]
    
    embeddings = (xlarge_embeddings + crepe_embeddings) / 2

    return embeddings, timestamps

def get_scene_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tensor:
    embeddings, _ = get_timestamp_embeddings(
        audio, model, hop_size_samples=SCENE_HOP_SIZE_SAMPLES
    )

    embeddings = torch.mean(embeddings, dim=1)
    return embeddings
