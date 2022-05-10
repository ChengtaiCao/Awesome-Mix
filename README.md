# Papers

## Bese Method
1.  Mixup: [Mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412) [2018; CV, Speech, Tabular, GAN, Adversarial Example]
2.  Framework: [Improved Mixed-Example Data Augmentation](https://arxiv.org/pdf/1805.11272.pdf?ref=https://githubhelp.com) [2018; CV]

## Variants
1.  CutMix: [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf) [2019; CV, Adversarial Example, Uncertainty]
2.  RICAP: [RICAP: Random Image Cropping and Patching Data Augmentation for Deep CNNs](http://proceedings.mlr.press/v95/takahashi18a/takahashi18a.pdf) [2018, CV]
3.  BC: [Learning from Between-Class Examples for Deep Sound Recognition](https://arxiv.org/pdf/1711.10282.pdf) [2018, Sound]
4.  Manifold Mixup: [Manifold Mixup: Better Representations by Interpolating Hidden States](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) [2019, CV]
5.  AdaMixUp: [MixUp as Locally Linear Out-of-Manifold Regularization](https://ojs.aaai.org/index.php/AAAI/article/download/4256/4134) [2019, CV]


## Others
1.  SamplePairing: [Data Augmentation by Pairing Samples for Images Classification](https://arxiv.org/pdf/1801.02929) [2018; CV]
2.  Cutout: [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf) [2018; CV]
3.  MixFeat: [Mixfeat: Mix feature in latent space learns discriminative space](https://openreview.net/pdf?id=HygT9oRqFX) [2019; CV]

## Applications
### CV
1.  BC+: [Between-class Learning for Image Classification](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tokozume_Between-Class_Learning_for_CVPR_2018_paper.pdf) (2018; CV)
### NLP
1.  wordMixup & senMixup: [Augmenting Data with Mixup for Sentence Classification: An Empirical Study](https://arxiv.org/pdf/1905.08941.pdf) [2019; NLP]
### Semi-Supervised
1.  ICT: [Interpolation Consistency Training for Semi-Supervised Learning](https://arxiv.org/pdf/1903.03825.pdf?ref=https://githubhelp.com) [2019; CV]
2.  MixMatch: [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://proceedings.neurips.cc/paper/2019/file/1cd138d0499a68f4bb72bee04bbec2d7-Paper.pdf) [2019; CV]
### Unsupervised: 
1.  ACAI: [Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer](https://arxiv.org/pdf/1807.07543.pdf) [2019; AE, GAN, CV]
2.  AMR: [On Adversarial Mixup Resynthesis](https://proceedings.neurips.cc/paper/2019/file/f708f064faaf32a43e4d3c784e6af9ea-Paper.pdf) [2019; AE, GAN, CV]
### Domain Adaptation 
1.  VMT: [Virtual Mixup Training for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1905.04215.pdf) [2019; CV]
2.  DMRL: [Dual Mixup Regularized Learning for Adversarial Domain Adaptation](https://arxiv.org/pdf/2007.03141) [2020; CV]
3.  IIMT: [Improve Unsupervised Domain Adaptation with Mixup Training](https://arxiv.org/pdf/2001.00677) [2022; CV]


# Code Repositories
{_Method_}\* denotes the code of _Method_ is reimplemented by others.
1. Mixip: https://github.com/facebookresearch/mixup-cifar10
2. Framework: https://github.com/ceciliaresearch/MixedExample
3. CutMix: https://github.com/clovaai/CutMix-PyTorch
4. RICAP*: https://github.com/4uiiurz1/pytorch-ricap
5. BC: https://github.com/mil-tokyo/bc_learning_sound
6. BC+: https://github.com/mil-tokyo/bc_learning_image
7. Manifold Mixup: https://github.com/vikasverma1077/manifold_mixup
8. ICT: https://github.com/vikasverma1077/ICT
9. AdaMixUp*: https://github.com/SITE5039/AdaMixUp
10. Cutout: https://github.com/uoguelph-mlrg/Cutout
11. AMR: https://github.com/christopher-beckham/amr
12. MixMatch: https://github.com/google-research/mixmatch
13. ACAI: https://github.com/brain-research/acai
14. VMT: https://github.com/xudonmao/VMT
15. DMRL: https://github.com/YuanWu3/Dual-Mixup-Regularized-Learning-for-Adversarial-Domain-Adaptation

## Keys
1.  Number of samples for mixup == 2;
2.  Data Augment v.s. Regularization --> Mixup v.s. Weight Decay;
3.  Why and which specific properties;
4.  Limits;
5.  Order of data augmentation;