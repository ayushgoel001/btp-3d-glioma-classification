# Primary references

These references support the data description and methods used in the report. This is the cited reference list for the completed baseline, not a claim of a systematic literature review.

1. **BraTS 2020 data documentation.** CBICA, University of Pennsylvania. [Official data page](https://www.med.upenn.edu/cbica/brats2020/data.html). Data organization, MRI modalities and organizer preprocessing.
2. **Menze, B. H., et al. (2015).** The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). *IEEE Transactions on Medical Imaging*, 34(10), 1993–2024. [DOI: 10.1109/TMI.2014.2377694](https://doi.org/10.1109/TMI.2014.2377694). Benchmark foundation.
3. **Bakas, S., et al. (2017).** Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features. *Scientific Data*, 4, 170117. [DOI: 10.1038/sdata.2017.117](https://doi.org/10.1038/sdata.2017.117). Expert annotations and collection context.
4. **He, K., Zhang, X., Ren, S., and Sun, J. (2016).** Deep Residual Learning for Image Recognition. *CVPR*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385). Residual learning used in the 3D adaptation.
5. **Wu, Y., and He, K. (2018).** Group Normalization. *ECCV*. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494). Normalization independent of batch statistics.
6. **Loshchilov, I., and Hutter, F. (2019).** Decoupled Weight Decay Regularization. *ICLR*. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101). AdamW optimization.
7. **PyTorch contributors.** CrossEntropyLoss, version 2.11 documentation. [Official reference](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html). Weighted mean reduction underlying the singleton-batch audit.

All numerical project scores derive from the saved Colab outputs in `results/gpu/`. Published numbers from different tasks and cohorts are not used as directly comparable baselines.
