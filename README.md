# ST2Vec

<div align=center>
<img src=./fig/framework.jpg width="80%" ></img>
</div>


This is our Pytorch implementation for the paper:

> Ziquan Fang, Yuntao Du, Xinjun Zhu, Danlei Hu, Lu Chen, Yunjun Gao and Christian S. Jensen. (2022). Spatio-Temporal Trajectory Similarity Learning in Road Networks. Paper in [ACM DL](https://dl.acm.org/doi/abs/10.1145/3534678.3539375) or Paper in [arXiv](https://arxiv.org/abs/2112.09339). In KDD'22, Washington DC, USA, August 14-18, 2022.

## Introduction

ST2Vec is a representation learning based solution that considers fine-grained spatial and temporal relations between trajectories to enable spatio-temporal similarity computation in road networks. 

## Citation

If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{ST2Vec22,
  author    = {Ziquan Fang and
               Yuntao Du and
               Xinjun Zhu and
               Danlei Hu and 
               Lu Chen and 
               Yunjun Gao and
               Christian S. Jensen},
  title     = {Spatio-Temporal Trajectory Similarity Learning in Road Networks},
  booktitle = {{KDD}},
  pages = {347–356},
  year      = {2022}
}
```

## Requirements

- Ubuntu OS
- Python >= 3.5 (Anaconda3 is recommended)
- PyTorch 1.4+
- A Nvidia GPU with cuda 10.2+

## Datasets

* Trajectory dataset (TDrive) and Rome are an open source data set
* We provided the road network data and map-matching result data

## Reproducibility & Training

1. Data preprocessing (Time embedding and node embedding)

   ```shell
   python preprocess.py
   ```

2. Ground truth generating (It will take a while...)

   ```shell
   python spatial_similarity.py
   python temporal_similarity.py
   ```

3. Triplets generating

   ```shell
   python data_utils.py
   ```

4. Training

   ```shell
   python main.py
   ```
