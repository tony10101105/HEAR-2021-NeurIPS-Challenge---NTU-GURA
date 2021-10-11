from typing import Tuple
import torch
from avgbert import hubert_wav2vec2
from torch import Tensor


def load_model(model_file_path: str = "") -> torch.nn.Module:

    model = hubert_wav2vec2()

    if torch.cuda.is_available():
        model.cuda()

    model.sample_rate = 16000
    model.embedding_size = 1024
    model.scene_embedding_size = model.embedding_size
    model.timestamp_embedding_size = model.embedding_size

    print("finish loading model")

    return model


def get_timestamp_embeddings(audio: Tensor, model: torch.nn.Module,) -> Tuple[Tensor, Tensor]:

    if audio.ndim != 2:
        raise ValueError(
            "audio input tensor must be 2D with shape (n_sounds, num_samples)"
        )

    print("getting timestamp model")

    model.eval()
    with torch.no_grad():
        embeddings = model(audio)

    audio_ms = int(audio.shape[1] / model.sample_rate * 1000)

    ntimestamps = (audio_ms - 5) // 20
    last_center = 12.5 + (ntimestamps - 1) * 20
    timestamps = torch.arange(12.5, last_center + 20, 20)
    assert len(timestamps) == ntimestamps
    timestamps = timestamps.expand((embeddings.shape[0], timestamps.shape[0]))
    assert timestamps.shape[1] == embeddings.shape[1]

    return embeddings, timestamps

def get_scene_embeddings(audio: Tensor, model: torch.nn.Module,) -> Tensor:

    print("getting scene embeddings")

    embeddings, _ = get_timestamp_embeddings(audio, model)
    embeddings = torch.mean(embeddings, dim=1)

    return embeddings
