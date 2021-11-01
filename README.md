# HEAR-2021-NeurIPS-Challenge---NTU

## Description

- We try many methods (cat, average, fusion) on three models (Hubert, Wav2vec2, Torchcrepe).
- Notice that Hubert models contain xlarge and large, Wav2vec2 model contains only large.
- The pretrained models me use are:
  - facebook/hubert-large-ll60k
  - facebook/hubert-xlarge-ll60k
  - facebook/wav2vec2-large-960h-lv60-self
  - torchcrepe
  
## Installation of the package

```shell
pip install \
git+https://github.com/tony10101105/HEAR-2021-NeurIPS-Challenge---NTU.git
```

## Usage

```python3
# In python code:
import GURA
```

## CUDA Version

* CUDA: 11.4

## Transformer Version

* 4.11.3

## Torchcrepe Version

* 0.0.15
