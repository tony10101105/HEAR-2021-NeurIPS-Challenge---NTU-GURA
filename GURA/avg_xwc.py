from typing import Tuple

import torch
from torch import Tensor

####### Crepe #######
import torchcrepe
SAMPLE_RATE = 16000

TIMESTAMP_HOP_SIZE = 50
SCENE_HOP_SIZE = 250

TIMESTAMP_HOP_SIZE_SAMPLES = (SAMPLE_RATE * TIMESTAMP_HOP_SIZE) // 1000
SCENE_HOP_SIZE_SAMPLES = (SAMPLE_RATE * SCENE_HOP_SIZE) // 1000

####### Hubert and Wav2vec2 #######
from transformers import Wav2Vec2Model, HubertModel


# hubert_xlarge: [batch_size, 49, 1280]
# wav2vec2: [batch_size, 49, 1024]
# torchcrepe: [batch_size, 5, 2048]

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

class XWC(torch.nn.Module):

    sample_rate = SAMPLE_RATE

    scene_embedding_size = 1280 + 1024 + 2048
    timestamp_embedding_size = 1024

    def __init__(self):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-xlarge-ll60k")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        self.crepe = TorchCrepeModel()

def load_model(model_file_path: str = "") -> torch.nn.Module:
    return XWC()

def get_timestamp_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
    hop_size_samples: int = TIMESTAMP_HOP_SIZE_SAMPLES,
) -> Tuple[Tensor, Tensor]:

    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (n_sounds, num_samples)"
        )

    if not isinstance(model, XWC):
        raise ValueError(f"Model must be an instance of {XWC.__name__}")

    # Send the model to the same device that the audio tensor is on.
    model.eval()
    with torch.no_grad():
        xlarge_embeddings = model.hubert(audio).last_hidden_state
        wav2vec2_embeddings = model.wav2vec2(audio).last_hidden_state
        crepe_embeddings = model.crepe(audio, hop_size_samples)

    def get_xlarge(embeddings):
        """
        same processing as hearbaseline
        """
        audio_ms = int(audio.shape[1] / model.sample_rate * 1000)
        ntimestamps = (audio_ms - 5) // 20
        last_center = 12.5 + (ntimestamps - 1) * 20
        timestamps = torch.arange(12.5, last_center + 20, 20)
        assert len(timestamps) == ntimestamps
        timestamps = timestamps.expand((embeddings.shape[0], timestamps.shape[0]))
        assert timestamps.shape[1] == embeddings.shape[1]
        # [batch_size, 49, 1280]
        return embeddings, timestamps

    def get_wav2vec2(embeddings):
        """
        same processing as hearbaseline
        """
        audio_ms = int(audio.shape[1] / model.sample_rate * 1000)
        ntimestamps = (audio_ms - 5) // 20
        last_center = 12.5 + (ntimestamps - 1) * 20
        timestamps = torch.arange(12.5, last_center + 20, 20)
        assert len(timestamps) == ntimestamps
        timestamps = timestamps.expand((embeddings.shape[0], timestamps.shape[0]))
        assert timestamps.shape[1] == embeddings.shape[1]
        # [batch_size, 49, 1024]
        return embeddings, timestamps

    def get_crepe(embeddings):
        """
        same processing as hearbaseline
        """
        ntimestamps = audio.shape[1] // hop_size_samples + 1
        hop_size = hop_size_samples * 1000 // SAMPLE_RATE

        timestamps = torch.tensor(
            [i * hop_size for i in range(ntimestamps)], device=embeddings.device
        )
        assert len(timestamps) == ntimestamps
        timestamps = timestamps.expand((embeddings.shape[0], timestamps.shape[0]))
        assert (
            timestamps.shape[1] == embeddings.shape[1]
        ), f"{timestamps.shape} vs {embeddings.shape}"
        # [batch_size, 5, 2048]
        return embeddings, timestamps

    xlarge_embeddings, xlarge_timestamps = get_xlarge(xlarge_embeddings)
    wav2vec2_embeddings, wav2vec2_timestamps = get_wav2vec2(wav2vec2_embeddings)
    crepe_embeddings, crepe_timestamps = get_crepe(crepe_embeddings)

    # [batch_size, {49, 49, 5}, {1280, 1024, 2048}] -> [batch_size, {1280, 1024, 2048}]
    xlarge_embeddings = torch.mean(xlarge_embeddings, dim=1)
    wav2vec2_embeddings = torch.mean(wav2vec2_embeddings, dim=1)
    crepe_embeddings = torch.mean(crepe_embeddings, dim=1)

    def compress_xlarge(embeddings):
        device = embeddings.device
        embeddings = embeddings.tolist()
        new_embeddings = []
        for embedding in embeddings:
            new_embedding = []
            for i in range(0, 1280, 5):
                for j in range(i, i + 4):
                    new_embedding.append((embedding[j] + embedding[j + 1]) / 2)
            new_embeddings.append(new_embedding)
        return torch.tensor(new_embeddings, device = device)

    def compress_crepe(embeddings):
        device = embeddings.device
        embeddings = embeddings.tolist()
        new_embeddings = []
        for embedding in embeddings:
            new_embeddings.append([(embedding[i] + embedding[i + 1]) / 2 for i in range(0, 2048, 2)])
        return torch.tensor(new_embeddings, device = device)

    xlarge_embeddings = compress_xlarge(xlarge_embeddings)
    crepe_embeddings = compress_crepe(crepe_embeddings)

    dim0, dim1 = xlarge_embeddings.shape[0], 1

    return torch.cat((xlarge_embeddings, wav2vec2_embeddings, crepe_embeddings), dim = 1), torch.randn(dim0, dim1)

def get_scene_embeddings(
    audio: Tensor,
    model: torch.nn.Module,
) -> Tensor:
    embeddings, _ = get_timestamp_embeddings(
        audio, model, hop_size_samples=SCENE_HOP_SIZE_SAMPLES
    )

    # not use timestamps here
    # already compress each embeddings to 1024 dimension

    xlarge_embeddings = embeddings[:, :1024]
    wav2vec2_embeddings = embeddings[:, 1024: 2048]
    crepe_embeddings = embeddings[:, 2048:]

    if (xlarge_embeddings.shape != crepe_embeddings.shape) or (crepe_embeddings.shape != wav2vec2_embeddings.shape):
        raise(f"wrong shape w-shape: {wav2vec2_embeddings.shape}, c-shape: {crepe_embeddings.shape}, x-shape: {xlarge_embeddings.shape}")

    return (xlarge_embeddings + crepe_embeddings + wav2vec2_embeddings) / 3
