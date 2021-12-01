# SSMN
![](https://img.shields.io/badge/language-python-orange.svg)
[![](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/fyancy/MetaFD/blob/main/LICENSE)
[![](https://img.shields.io/badge/CSDN-燕策西-blue.svg)](https://blog.csdn.net/weixin_43543177?spm=1001.2101.3001.5343)
<!-- 如何设计GitHub badge: https://lpd-ios.github.io/2017/05/03/GitHub-Badge-Introduction/ -->

The PyTorch implementation for `semi-supervised meta-learning networks (SSMN)` in [Semi-supervised meta-learning networks with squeeze-and-excitation attention for few-shot fault diagnosis](https://www.sciencedirect.com/science/article/pii/S0019057821001543).

## Abstract
In the engineering practice, lacking of data especially labeled data typically hinders the wide application of deep learning in mechanical fault diagnosis. However, collecting and labeling data is often expensive and time-consuming. To address this problem, a kind of semi-supervised meta-learning networks (SSMN) with squeeze-and-excitation attention is proposed for few-shot fault diagnosis in this paper. SSMN consists of a parameterized encoder, a non-parameterized prototype refinement process and a distance function. Based on attention mechanism, the encoder is able to extract distinct features to generate prototypes and enhance the identification accuracy. With semi-supervised few-shot learning, SSMN utilizes unlabeled data to refine original prototypes for better fault recognition. A combinatorial learning optimizer is designed to optimize SSMN efficiently. The effectiveness of the proposed method is demonstrated through three bearing vibration datasets and the results indicate the outstanding adaptability in different situations. Comparison with other approaches is also made under the same setup and the experimental results prove the superiority of the proposed method for few-shot fault diagnosis.

## Citation
```
@article{FENG2021,
title = {Semi-supervised meta-learning networks with squeeze-and-excitation attention for few-shot fault diagnosis},
journal = {ISA Transactions},
year = {2021},
issn = {0019-0578},
doi = {https://doi.org/10.1016/j.isatra.2021.03.013},
url = {https://www.sciencedirect.com/science/article/pii/S0019057821001543},
author = {Yong Feng and Jinglong Chen and Tianci Zhang and Shuilong He and Enyong Xu and Zitong Zhou},
keywords = {Fault diagnosis, Semi-supervised learning, Meta-learning, Attention, Few-shot classification}
```
