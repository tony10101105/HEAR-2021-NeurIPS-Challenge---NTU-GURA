# HEAR-2021-NeurIPS-Challenge---NTU

## Description

- We try many methods (cat, average, fusion) on three models (Hubert, Wav2vec2, Torchcrepe).
- We adopt two methods on the relationship between our scene-embedding and timestamp-embedding models. In "fusion_cat_xwc_time", every certain time inverted is averaged and concatenated. In other models, we simply average three models'(Hubert, Wav2vec2, Torchcrepe) embeddings.
- Notice that Hubert models contain xlarge and large, Wav2vec2 model contains only large.
- The pretrained models we use are:
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

```python
# In python code:
from GURA import fusion_wav2vec2
from GURA import cat_wc
.
.
.
```

## Python Version

* python3.8

## CUDA Version

* CUDA: 11.4

## Torch Version
* torch: 1.9.1+cu111

## Transformer Version

* 4.11.3

## Torchcrepe Version

* 0.0.15
