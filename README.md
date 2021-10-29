# HEAR-2021-NeurIPS-Challenge---NTU

## Description

We concatenate the embeddings generated from the three models below.

```
1. HuBert Xlarge (without fusion)
2. Wav2Vec2 (without fusion)
3. Crepe (without fusion)
```

## Installation of the package

```shell
pip install \
git+https://github.com/tony10101105/HEAR-2021-NeurIPS-Challenge---NTU.git@cat_hubert_wav2vec2_crepe
```

## Usage

```python3
# In python code:
from GURA import cat_xwc
```
## CUDA Version

* CUDA: 11.4

## Package Version

* torch==1.9.1+cu111
* transformers==4.11.3
