# HEAR-2021-NeurIPS-Challenge---NTU

## Description

We concatenate the embeddings generated from the three models below.
```
1. HuBert Xlarge (with fusion)
2. Wav2Vec2 (with fusion)
3. Crepe (with fusion)
```
* Make use of timestamps
* Get more scene_embeddings by splitting time_embeddings
## Installation of the package

```shell
pip install \
git+https://github.com/tony10101105/HEAR-2021-NeurIPS-Challenge---NTU.git@fusion_cat_xwc_time
```

## Usage

```python3
# In python code:
from GURA import fusion_cat_xwc_time
```
## CUDA Version

* CUDA: 11.4

## Package Version

* torch==1.9.1+cu111
* transformers==4.11.3
