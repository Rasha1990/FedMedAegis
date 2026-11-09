# Federated Medical Imaging with Annotation-Free Lesion Localization and Pathology-Aware Risk Prioritization

This repository contains the official implementation of **FedMedAegis**, a unified federated learning framework for medical imaging. FedMedAegis addresses critical challenges in decentralized healthcare AI by including: (1) clinical risk-aware aggregation to prioritize high-risk pathologies, (2) Byzantine-robust defense against adversarial clients, (3) calibrated differential privacy for regulatory compliance, and (4) weakly supervised lesion localization without dense annotations—enabling secure, auditable, and parameter-efficient collaborative learning across hospitals.

> **Note on Code Availability:** 
> The complete source code, training scripts, and detailed preprocessing pipelines will be made publicly available upon the official acceptance of our manuscript. For the duration of the peer-review process, the full codebase has been provided to the journal editors and reviewers as part of the submission package to ensure transparency and reproducibility.

## Dataset Information
FedMedAegis was evaluated on five publicly available medical imaging datasets. We do not host the raw data in this repository due to size and licensing constraints, but the official sources are listed below. Preprocessing and stratified splitting scripts are included in the `scripts/` directory.

1. **NIH ChestX-ray14**: [Official NIH Clinical Center Repository](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345)
2. **ISIC 2018 (Task 3)**: [Official ISIC Archive](https://challenge.isic-archive.com/data/)
3. **PH2 Dataset**: [PH2 Database via Kaggle](https://www.kaggle.com/datasets/spacesurfer/ph2-dataset)
4. **HAM10000**: [HAM10000 via Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
5. **Derm7pt**: [Derm7pt via Kaggle](https://www.kaggle.com/datasets/muhammedfirat/derm7pt)

*Note: As detailed in the manuscript, HAM10000 serves as the initial training cohort for ISIC 2018 Task 3. In our experiments, they are treated as distinct experimental groups based on their metadata structures and client partitioning strategies.*

## Framework Overview
The FedMedAegis framework addresses the interdependent challenges of data privacy, heterogeneity, and diagnostic prioritization in federated medical image analysis. Key components include:
- **Adaptive Federated Averaging**: Reputation-weighted aggregation combined with Krum-based geometric filtering to mitigate Byzantine attacks and free-rider behavior.
- **FALS-DGBR Module**: A lightweight, frequency-aware segmentation architecture that localizes lesions without requiring pixel-level annotations.
- **DAG-Based Auditability**: Immutable, round-by-round logging of client performance and aggregation weights to satisfy regulatory standards (e.g., FDA 21 CFR Part 11).

## Reproduction and Setup
This repository includes:
- A `requirements.txt` file for dependency management.
- Step-by-step scripts for data preprocessing, non-IID client partitioning, and model training in the `scripts/` directory.
- Configuration files matching the hyperparameters reported in the manuscript in the `configs/` directory.

## Citation
If you find this work or the provided code useful for your research, please cite our paper:

```bibtex
@article{alhuthaifi2026fedmedaegis,
  title={Risk-Aware Differentially Private Federated Learning with Weakly Supervised Lesion Localization for Medical Imaging},
  author={Al-Huthaifi, Rasha and Yang, Hailiang and Wang, Hengzhi and Cui, Laizhong and Huang, Wei and Al-Huda, Zaid},
  journal={Expert Systems with Applications},
  year={2026},
  publisher={Elsevier}
}
