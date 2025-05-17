# Introduction
This is the implementation of our paper: Hybrid Batch Normalisation: Resolving the Dilemma of Batch Normalisation in Federated Learning (accepted by ICML 2025 poster).
<img src="./adaptive_normalisation.png" alt="adaptive_normalisation">
**(a) Original Clusters**:
Cluster 1 (blue squares): 300 samples centered at [5, 5] with a tight standard deviation of  0.4; 
Cluster 2 (coral circles): 200 samples centered at [-1, -1] with a wider standard deviation of 1.2.

**(b) Local Normalistion**:
Each cluster is normalised independently using its own local statistics. 

**(c) Global Normalistion**:
Both clusters are normalised using the global statistics. 

**(d) Hybrid Normalistion**:
Each cluster is normalised by a specific mixture of local and global statistics. 
Cluster 1: 70% reliance on local statistics, 30% on global statistics; Cluster 2: 30% reliance on local statistics, 70% on global statistics

**Hybrid normalisation combines global statistics with local statistics, which can standardise the size of two clusters while maintaining the global structure.**

# Hybrid Batch Normalisation
<img src="./HBN.png" alt="HybridBN" width="650">
The specific code implementation can be found in <a href="./FedBaseline/models/FedHBN/MyBNtool.py" target="_blank" title="HBN">./FedBaseline/models/FedHBN/MyBNtool.py</a>.

# File Structure （Updating）
```text
├── FedBaseline/
│   ├── models/
│   │   ├── FedAvg/          implementation of client and server operations for FedAvg
│   │   ├── FBN/             implementation of client and server operations for FedAvg+FBN
│   │   ├── FedFN/           implementation of client and server operations for FedAvg+FedFN
│   │   ├── FixBN/           implementation of client and server operations for FedAvg+FixBN
│   │   ├── FedHBN/          implementation of client and server operations for FedAvg+HBN
│   │   ├── Net.py           storage network architecture.
│   │   └── Test.py          used for testing.
│   ├── options/
│   │   └── options.py       experimental parameter settings.
│   └── sampling/
│       ├── dataloader.py    load data.
│       └── sampling.py      divide the data to the clients.
└── XXX_main.py              core driver program for specific algorithms.
```
FedAvg:[Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html) *AISTATS 2017*

FBN:[Overcoming the Challenges of Batch Normalization in Federated Learning](https://arxiv.org/abs/2405.14670) arXiv

FedFN: [FedFN: Feature Normalization for Alleviating Data Heterogeneity Problem in Federated Learning](https://openreview.net/forum?id=4apX9Kcxie) *FL@FM-NeurIPS’23*

FixBN: [Making Batch Normalization Great in Federated Deep Learning](https://openreview.net/forum?id=iKQC652XIk) *FL@FM-NeurIPS’23*

# Runing
**Note**: Do not run all methods with the same initial learning rate, as different normalisation methods may have different optimal learning rates. Some tuning experience can be found in the appendix of the paper. 

# Citation

