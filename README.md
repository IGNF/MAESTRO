# MAESTRO: Masked Autoencoders for Multimodal, Multitemporal, and Multispectral Earth Observation Data

## Abstract
We introduce **MAESTRO**, a tailored adaptation of the Masked Autoencoder (MAE) framework that effectively orchestrates the use of multimodal, multitemporal, and multispectral Earth Observation (EO) data. Evaluated on four EO datasets, MAESTRO sets a new state-of-the-art on tasks that strongly rely on multitemporal dynamics, while remaining highly competitive on others.

<p align="center">
  <img src="img/Maestro_Overview.png" alt="MAESTRO Overview" width="750"/><br>
  <em>Figure 1 — MAESTRO overview.</em>
</p>

## Contributions

- **Extensive Benchmarking of Multimodal/Multitemporal SSL:** Impact evaluation of various fusion strategies for multimodal and multitemporal SSL.
- **Novel Patch-group-wise Normalization:** Novel normalization scheme that normalizes reconstruction targets patch-wise within groups of highly correlated spectral bands.
- **MAESTRO:** Novel adaptation of the MAE that combines optimized fusion strategies with our tailored patch-group-wise normalization.

<p align="center">
  <img src="img/Maestro_Fusion.png" alt="MAESTRO fusion modes" width="750"/><br>
  <em>Figure 2 — Token-based fusion modes benchmarked for multimodal and multitemporal SSL.</em>
</p>

## Results

<p align="center">
  <em>
    Table 1 — Performance comparison of MAESTRO, baseline FMs, and supervised ViTs.<br>
    We report weighted F1 score (%) on TreeSatAI-TS and mIoU (%) on PASTIS-HD, FLAIR#2, and FLAIR-HUB.<br>
    MAESTRO† models were pre-trained for twice the number of epochs.
  </em>

| Model              | Size  | Fusion mode   | TreeSatAI-TS | PASTIS-HD | FLAIR#2 | FLAIR-HUB |
|--------------------|-------|---------------|--------------|-----------|---------|-----------|
| **MAESTRO (ours)** |       |               |              |           |         |           |
| MAESTRO            | Base  | group         | 78.5               | 68.8                | 63.8                | 64.9               |
| MAESTRO            | Base  | inter-group   | 78.8               | 68.6                | 62.6                | **65.9** 🔴↓**0.1**|
| MAESTRO†           | Base  | group         | 79.1               | 68.8                | **64.0** 🔴↓**0.2** | 64.8               |
| MAESTRO†           | Base  | inter-group   | **79.4** 🟢↑**2.7**| **69.0** 🟢↑**2.5** | 63.3                | 65.8               |
| **Baseline FMs**   |       |               |              |           |         |           |
| DINO-v2            | Base  | shared        | **76.7**     | 64.4      | **64.2**| **66.0**  |
| DINO-v2 sat.       | Large | shared        | 76.3         | 64.0      | 63.5    | **66.0**  |
| DOFA               | Base  | shared        | 76.0         | 62.9      | 62.3    | 65.1      |
| CROMA              | Base  | inter-croma   | 70.5         | 65.0      | 39.0    | 44.3      |
| AnySat             | Base  |               | 75.1         | **66.5**  | 55.1    |           |
| **Supervised ViTs**|       |               |              |           |         |           |
| ViT                | Base  | group         | **75.7**     | **64.6**  | **58.3**| 61.6      |
| ViT                | Base  | inter-group   | 75.6         | 64.5      | 58.2    | **62.1**  |
| **Previous SOTA**  |       |               |              |           |         |           |
|                    |       |               | 75.1         | 66.5      | 64.1    | 64.3  |

</p>



### Datasets
Our implementation already supports 4 datasets:

**[TreeSatAI-TS](https://huggingface.co/datasets/IGNF/TreeSatAI-Time-Series)**
Tree species identification, with 15 multi-label classes.
  - Extent: 50,381 tiles of 60 × 60 m in Germany.
  - Modalities: aerial imagery RGB + NIR (0.2 m), Sentinel-1 time series, Sentinel-2 time series.

**[PASTIS-HD](https://huggingface.co/datasets/IGNF/PASTIS-HD)**
Agricultural crop segmentation, with 19 semantic classes.
  - Extent: 433 tiles of 1280 × 1280 m in France.
  - Modalities: VHR satellite imagery SPOT 6-7 (1 m), Sentinel-1 time series, Sentinel-2 time series.

**[FLAIR#2](https://arxiv.org/abs/2305.14467)**
Land cover segmentation, with 12 semantic classes.
  - Extent: 77,762 tiles of 102.4 × 102.4 m in France.
  - Modalities: Aerial and elevation imagery RGB + NIR + DEM + DSM (0.2 m), Sentinel-2 time series.

**[FLAIR-HUB](https://huggingface.co/datasets/IGNF/FLAIR-HUB)**
Land cover segmentation, with 15 semantic classes.
  - Extent: 241,100 tiles of 102.4 × 102.4 m in France.
  - Modalities: Aerial and elevation imagery RGB + NIR + DEM +
DSM (0.2 m), Sentinel-1 time series, Sentinel-2 time series.

Note that our version of FLAIR#2 is not the original release but a regenerated version derived through the refiltering of FLAIR-HUB.

## Getting Started
```bash
# 1. Change directory
cd MAESTRO

# 2. Install dependencies with Poetry
poetry install
```

### Minimal examples

You can start from the following minimal examples:

On TreeSatAI-TS:
```bash
poetry run python main.py \
        model.model=mae \
        model.model_size=medium \
        run.exp_name=mae-m_treesat \
        run.exp_dir=/path/to/experiments/dir \
        datasets.root_dir=/path/to/dataset/dir \
        datasets.treesatai_ts.rel_dir=TreeSatAI-TS \
        datasets.filter_pretrain=[treesatai_ts] \
        datasets.filter_finetune=[treesatai_ts]
```

On PASTIS-HD:
```bash
poetry run python main.py \
        model.model=mae \
        model.model_size=medium \
        run.exp_name=mae-m_pastis \
        run.exp_dir=/path/to/experiments/dir \
        datasets.root_dir=/path/to/dataset/dir \
        datasets.pastis_hd.rel_dir=PASTIS-HD \
        datasets.filter_pretrain=[pastis_hd] \
        datasets.filter_finetune=[pastis_hd]
```

On FLAIR-HUB:
```bash
poetry run python main.py \
        model.model=mae \
        model.model_size=medium \
        run.exp_name=mae-m_flair \
        run.exp_dir=/path/to/experiments/dir \
        datasets.root_dir=/path/to/dataset/dir \
        datasets.flair.rel_dir=FLAIR-HUB \
        datasets.filter_pretrain=[flair] \
        datasets.filter_finetune=[flair]
```

Most hyperparameters can be adapted through the hydra-zen CLI.


## Contact

For questions or contributions, please open an issue or contact the authors.


## Reference

If you use this code, kindly cite:

```bibtex
@article{labatie2025maestro,
  title={MAESTRO: Masked AutoEncoders for Multimodal, Multitemporal, and Multispectral Earth Observation Data},
  author={Labatie, Antoine and Vaccaro, Michael and Lardiere, Nina and Garioud, Anatol and Gonthier, Nicolas},
  journal={arXiv preprint arXiv:2508.10894},
  year={2025}
}
```

## Acknowledgement

The experiments in the paper were conducted using HPC/AI resources from GENCI-IDRIS (allocations A0181013803, A0161013803, and AD010114597R1).
