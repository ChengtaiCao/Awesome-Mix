# Bese Method
* [Mixup -- ICLR 2018] [Mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412) [[Code](https://github.com/facebookresearch/mixup-cifar10)]
* [Cutmix -- ICCV 2019] [CutMix: Regularization Strategy to Train Strong Classifiers With Localizable Features](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf) [[Code](https://github.com/clovaai/CutMix-PyTorch)]

# Theory
## Vicinal Risk Minimization (VRM)
* [VRM -- NeurIPS 2000] [Vicinal Risk Minimization](https://proceedings.neurips.cc/paper/2000/file/ba9a56ce0a9bfa26e8ed9e10b2cc8f46-Paper.pdf)
* [VRM -- ICLR 2022] [Towards Understanding the Data Dependency of Mixup-style Training](https://arxiv.org/pdf/2110.07647) [[code](https://github.com/2014mchidamb/Mixup-Data-Dependency)]
## Regularization
* [Regularization -- Arxiv 2020] [On Mixup Regularization](http://arxiv.org/pdf/2006.06049)
* [Regularization -- ICLR 2021] [How Does Mixup Help With Robustness and Generalization?](http://arxiv.org/pdf/2010.04819) 
* [Regularization -- Arxiv 2022] [A Unified Analysis of Mixed Sample Data Augmentation: A Loss Function Perspective](http://arxiv.org/pdf/2208.09913) [[code](https://github.com/naver-ai/hmix-gmix)]
* [Regularization -- IEEE Access 2018] [Understanding mixup training methods](https://ieeexplore.ieee.org/iel7/6287639/8274985/08478159.pdf) [[code](https://github.com/liangdaojun/spatial-mixup)]
## Adversarial Training
* [Adversarial Training -- Arxiv 2020] [FMix: Enhancing Mixed Sample Data Augmentation](http://arxiv.org/pdf/2002.12047) [[code](https://github.com/ecs-vlc/FMix)]
* [Adversarial Training -- Arxiv 2019] [MixUp as Directional Adversarial Training](http://arxiv.org/pdf/1906.06875)
## Uncertainty \& Calibration
* [Uncertainty \& Calibration -- NIPS 2019] [On Mixup Training: Improved Calibration and Predictive Uncertainty for Deep Neural Networks](https://proceedings.neurips.cc/paper/2019/file/36ad8b5f42db492827016448975cc22d-Paper.pdf)
* [Uncertainty \& Calibration -- ICML 2022] [When and How Mixup Improves Calibration](https://proceedings.mlr.press/v162/zhang22f/zhang22f.pdf)

# Variant
## Mixup-Based
* [BC Learning -- ICLR 2018] [Learning from Between-class Examples for Deep Sound Recognition](https://arxiv.org/pdf/1711.10282) [[code](https://github.com/mil-tokyo/bc_learning_sound)]
* [BC Learning + -- CVRP 2018] [Between-class Learning for Image Classification](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tokozume_Between-Class_Learning_for_CVPR_2018_paper.pdf) [[code](https://github.com/mil-tokyo/bc_learning_image)]
* [SamplePairing -- Arxiv 2018] [Data Augmentation by Pairing Samples for Images Classification](https://arxiv.org/pdf/1801.02929) [[_code_\*](https://github.com/junkwhinger/SamplePairing)]
* [Manifold Mixup -- ICML 2019] [Manifold Mixup: Better Representations by Interpolating Hidden States](http://proceedings.mlr.press/v97/verma19a/verma19a.pdf) [[_code_\*](https://github.com/vikasverma1077/manifold_mixup)]
* [AlignMixup -- CVPR 2022] [AlignMixup: Improving Representations By Interpolating Aligned Features](https://openaccess.thecvf.com/content/CVPR2022/papers/Venkataramanan_AlignMixup_Improving_Representations_by_Interpolating_Aligned_Features_CVPR_2022_paper.pdf) [[code](https://github.com/shashankvkt/AlignMixup_CVPR22)]
* [AdaMixUp -- AAAI 2019] [MixUp as Locally Linear Out-of-Manifold Regularization](https://ojs.aaai.org/index.php/AAAI/article/download/4256/4134) [[_code_\*](https://github.com/SITE5039/AdaMixUp)]
* [MetaMixUp -- TNNLS 2021] [MetaMixUp: Learning Adaptive Interpolation Policy of MixUp with Meta-Learning](https://ieeexplore.ieee.org/iel7/5962385/6104215/09366422.pdf?casa_token=yeyn9EGO8PAAAAAA:QxbqK3Y2lYbKN1eX-wOGg6rf99WUalLPE3sKzkSSuwEBmDCky3E3ozOiA5-BjYM75qmAd5EdwvA)
* [AutoMix -- ECCV 2022] [AutoMix: Unveiling the Power of Mixup for Stronger Classifiers](https://arxiv.org/pdf/2103.13027) [[code](https://github.com/Westlake-AI/openmixup)]
* [Local Mixup -- Arxiv 2022] [Preventing Manifold Intrusion with Locality: Local Mixup](https://arxiv.org/pdf/2201.04368) [[code](https://github.com/raphael-baena/Local-Mixup)]
* [CAMixup -- ICLR 2021] [Combining Ensembles and Data Augmentation can Harm your Calibration](https://arxiv.org/pdf/2010.09875) [[code](https://github.com/google/edward2/tree/master/experimental/marginalization_mixup)]
* [Pani -- Arxiv 2019] [Patch-level Neighborhood Interpolation: A General and Effective Graph-based Regularization Strategy](https://arxiv.org/pdf/1911.09307) 
* [LADA -- EMNLP 2020] [Local Additivity Based Data Augmentation for Semi-supervised NER](https://arxiv.org/pdf/2010.01677) [[code](https://github.com/GT-SALT/LADA)]
* [HypMix -- EMNLP 2021] [HYPMIX: Hyperbolic Interpolative Data Augmentation](https://aclanthology.org/2021.emnlp-main.776) [[code](https://github.com/caisa-lab/hypmix-emnlp)]
* [SAMix -- Arxiv 2021] [Boosting Discriminative Visual Representation Learning with Scenario-Agnostic Mixup](https://arxiv.org/pdf/2111.15454)
* [GenLabel -- ICML 2022] [GenLabel: Mixup Relabeling using Generative Models](https://arxiv.org/pdf/2201.02354) [[code](https://github.com/UW-Madison-Lee-Lab/GenLabel_official)]
* [DMix -- ACL 2022] [DMIX: Adaptive Distance-aware Interpolative Mixup](https://aclanthology.org/2022.acl-short.67.pdf) [[code](https://github.com/caisa-lab/DMix-ACL)]
* [Nonlinear Mixup -- AAAI 2020] : [Out-Of-Manifold Data Augmentation for Text Classification](https://ojs.aaai.org/index.php/AAAI/article/view/5822/5678)
* [AMP -- EMNLP 2021] [Adversarial Mixing Policy for Relaxing Locally Linear Constraints in Mixup](https://arxiv.org/pdf/2109.07177) [[code](https://github.com/PAI-SmallIsAllYourNeed/Mixup-AMP)]
* [MixMo -- CVPR 2021] [MixMo: Mixing Multiple Inputs for Multiple Outputs via Deep Subnetworks](https://openaccess.thecvf.com/content/ICCV2021/papers/Rame_MixMo_Mixing_Multiple_Inputs_for_Multiple_Outputs_via_Deep_Subnetworks_ICCV_2021_paper.pdf) [[code](https://github.com/alexrame/mixmo-pytorch)]
* [SuperMix -- CVPR 2021] [SuperMix: Supervising the Mixing Data Augmentation](http://openaccess.thecvf.com/content/CVPR2021/papers/Dabouei_SuperMix_Supervising_the_Mixing_Data_Augmentation_CVPR_2021_paper.pdf) [[code](https://github.com/alldbi/SuperMix)]
* [DJMix -- Arxiv 2021] [DJMix: Unsupervised Task-agnostic Augmentation for Improving Robustness](https://openreview.net/pdf?id=0n3BaVlNsHI)
* [StyleMix -- CVPR 2021] [StyleMix: Separating Content and Style for Enhanced Data Augmentation](https://openaccess.thecvf.com/content/CVPR2021/papers/Hong_StyleMix_Separating_Content_and_Style_for_Enhanced_Data_Augmentation_CVPR_2021_paper.pdf) [[code](https://github.com/alsdml/StyleMix)]
* [Mixstyle -- ICLR 2021] : [Domain generalization with mixstyle](https://arxiv.org/pdf/2104.02008) [[code](https://github.com/KaiyangZhou/mixstyle-release)]
* [MoEx -- CVPR 2021] [On Feature Normalization and Data Augmentation](http://openaccess.thecvf.com/content/CVPR2021/papers/Li_On_Feature_Normalization_and_Data_Augmentation_CVPR_2021_paper.pdf) [[code](https://github.com/Boyiliee/MoEx)]
* [DM -- Arxiv 2022] [Decoupled Mixup for Data-efficient Learning](https://arxiv.org/pdf/2203.10761)
* [Mixup-with-AUM-and-SM -- ACL 2022]: [On the Calibration of Pre-trained Language Models using Mixup Guided by Area Under the Margin and Saliency](https://arxiv.org/pdf/2203.07559) 
* [BatchMixup -- IJCNLP 2021] [BATCHMIXUP: Improving Training by Interpolating Hidden States of the Entire Mini-batch](https://aclanthology.org/2021.findings-acl.434.pdf)
* [EXA Mixup -- TKDD 2022] [Explainability-based mix-up approach for text data augmentation](https://dl.acm.org/doi/pdf/10.1145/3533048)
* [PixMix -- CVPR 2022] [PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures](https://openaccess.thecvf.com/content/CVPR2022/papers/Hendrycks_PixMix_Dreamlike_Pictures_Comprehensively_Improve_Safety_Measures_CVPR_2022_paper.pdf) [[code](https://github.com/andyzoujm/pixmix)]
* [AugMix -- ICLR 2020] [AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty](https://openreview.net/pdf?id=S1gmrxHFvB) [[code](https://github.com/google-research/augmix)]
* [k-Mixup -- Arxiv 2021] [k-Mixup Regularization for Deep Learning via Optimal Transport](https://arxiv.org/pdf/2106.02933) 
* [MultiMix -- Arxiv 2022] [Teach me how to Interpolate a Myriad of Embeddings](https://arxiv.org/pdf/2205.14230)
* [Remix -- ECCV 2022] [Remix: Rebalanced Mixup ](https://arxiv.org/pdf/2007.03943)
* [MWh -- ICIG 2021] [Mixup Without Hesitation](https://arxiv.org/pdf/2101.04342) [[code](https://github.com/yuhao318/mwh)]
* [AutoMix -- ECCV 2020] [AutoMix: Mixup Networks for Sample Interpolation via Cooperative Barycenter Learning](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550630.pdf) [[_code_\*](https://github.com/ReBoRn8888/AutoMix)]
* [RegMix -- Arxiv 2021] [RegMix: Data Mixing Augmentation for Regression](https://arxiv.org/pdf/2106.03374.pdf)
* [M-Mix -- KDD 2022] [M-Mix: Generating Hard Negatives via Multi-sample Mixing for Contrastive Learning](https://dl.acm.org/doi/pdf/10.1145/3534678.3539248?casa_token=4rYYzEtCcxUAAAAA:JHz8mo3l1W-1bV5kh_PVrdyaIhASZkASIAMI5n-ZZM8jAyVWw-o3CXNgsi9uZIrbQLQiAbhLoV-WV8w) [[code]( https://github.com/Sherrylone/m-mix)]
* [CSANMT -- ACL 2022] [Learning to Generalize to More: Continuous Semantic Augmentation for Neural Machine Translation](https://arxiv.org/pdf/2204.06812) [[code](https://github.com/pemywei/csanmt)]
* [RegMixup -- NIPS 2022] [RegMixup: Mixup as a Regularizer Can Surprisingly Improve Accuracy and Out Distribution Robustness](https://arxiv.org/pdf/2206.14502) [[code](https://github.com/FrancescoPinto/RegMixup)]`

## Cutmix-Based
* [MixdedExample -- WACV 2019] [Improved Mixed-Example Data Augmentation](https://ieeexplore.ieee.org/iel7/8642793/8658235/08659168.pdf?casa_token=vxPrsAdypIAAAAAA:3D8UWPSlFNhIpF7K9KKb3hSdDF79p3DXchPTv5qRBQHlJ8VwyDbldMUp0rtbxGVR5dDwBHMwfM8) [[code](https://github.com/ceciliaresearch/MixedExample)]
* [RICAP -- ACML 2018] [RICAP: Random Image Cropping and Patching Data Augmentation for Deep CNNs](http://proceedings.mlr.press/v95/takahashi18a/takahashi18a.pdf) [[_code_\*](https://github.com/4uiiurz1/pytorch-ricap)]
* [Attentive Cutmix -- ICASSP 2020] [Attentive CutMix: An Enhanced Data Augmentation Approach for Deep Learning Based Image Classification](https://arxiv.org/pdf/2003.13048) [[_code_\*](https://github.com/xden2331/attentive_cutmix)]
* [FocusMix -- ICTC 2020] [Where to Cut and Paste: Data Regularization with Selective Features](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9289404&casa_token=ZOkscThNTpQAAAAA:AhalGFG_kjrXgaEZRzo5E3QN2mNC7gdnF1PtAd7MG0-rXbaHSS1JzZiM5wWv7hLR8plKxXy4F3U)
* [TransMix -- CVPR 2022] [TransMix: Attend to Mix for Vision Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_TransMix_Attend_To_Mix_for_Vision_Transformers_CVPR_2022_paper.pdf) [[code](https://github.com/Beckschen/TransMix)]
* [SaliencyMix -- ILCR 2021] [SaliencyMix: A Saliency Guided Data Augmentation Strategy for Better Regularization](https://arxiv.org/pdf/2006.01791) [[code](https://github.com/SaliencyMix/SaliencyMix)]
* [Puzzle Mix -- ICML 2020] [Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup](http://proceedings.mlr.press/v119/kim20b/kim20b.pdf) [[code](https://github.com/snu-mllab/PuzzleMix)]
* [SSMix -- ACL 2021] [SSMix: Saliency-Based Span Mixup for Text Classification](https://arxiv.org/pdf/2106.08062) [[code](https://github.com/clovaai/ssmix)]
* [Attribute Mix -- VCIP 2020] [Attribute Mix: Semantic Data Augmentation for Fine Grained Recognition](https://arxiv.org/pdf/2004.02684) 
* [Saliency Grafting -- AAAI 2022] [Saliency Grafting: Innocuous Attribution-Guided Mixup with Calibrated Label Mixing](https://ojs.aaai.org/index.php/AAAI/article/view/20766/20525)
* [Co-Mixup -- ICLR 2021] [Co-Mixup: Saliency Guided Joint Mixup with Supermodular Diversity](https://arxiv.org/pdf/2102.03065) [[code](https://github.com/snu-mllab/Co-Mixup)]
* [SnapMix -- AAAI 2021] [SnapMix: Semantically Proportional Mixing for Augmenting Fine-grained Data](https://ojs.aaai.org/index.php/AAAI/article/view/16255/16062) [[code](https://github.com/Shaoli-Huang/SnapMix)]
* [ResizeMix -- Arxiv 2020] [ResizeMix: Mixing Data with Preserved Object Information and True Labels](https://arxiv.org/pdf/2012.11101.pdf) [[_code_\*](https://github.com/HarborYuan/pytorch-macos-bench)]
* [TokenMix -- ECCV 2022] [TokenMix: Rethinking Image Mixing for Data Augmentation in Vision Transformers](https://arxiv.org/pdf/2207.08409) [[code](https://github.com/Sense-X/TokenMix)]
* [ScoreMix -- CVPR 2022] [ScoreNet: Learning Non-Uniform Attention and Augmentation for Transformer-Based Histopathological Image Classification](https://arxiv.org/pdf/2202.07570) 
* [GridMix -- Pattern Recognition 2021] [GridMix: Strong regularization through local context mapping](https://www.sciencedirect.com/science/article/pii/S0031320320303976?casa_token=oQ7NhHPxs1cAAAAA:U0cFG2ASbufAHEPW4m14bxaUMsXK3QE6ke-sjpvbpkcbbLAd_YFSUEbUU2DECq3H7IjtW2dRpAQ)
* [PatchMix -- BMWC 2021] [Evolving Image Compositions for Feature Representation Learning](https://arxiv.org/pdf/2106.09011)
* [PatchUp -- AAAI 2022] [PatchUp: A Regularization Technique for Convolutional Neural Networks](https://arxiv.org/pdf/2006.07794) [[code](https://github.com/chandar-lab/PatchUp)]
* [ChessMix -- SIBGRAPI 2021] [ChessMix: Spatial Context Data Augmentation for Remote Sensing Semantic Segmentation](https://ieeexplore.ieee.org/iel7/9643073/9642965/09643145.pdf) [[code](https://github.com/matheusbarrosp/chessmix)]
* [RecursiveMix -- Arxiv 2022] [RecursiveMix: Mixed Learning with History](https://arxiv.org/pdf/2203.06844) [[code](https://github.com/megvii-research/RecursiveMix)]
* [SmoothMix -- CVPR 2020] [SmoothMix: a Simple Yet Effective Data Augmentation to Train Robust Classifiers](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w45/Lee_SmoothMix_A_Simple_Yet_Effective_Data_Augmentation_to_Train_Robust_CVPRW_2020_paper.pdf) [[code](https://github.com/jh-jeong/smoothmix)]
* [SuperpixelGridMix -- Arxiv 2022] [SuperpixelGridCut, SuperpixelGridMean and SuperpixelGridMix Data Augmentation](https://arxiv.org/pdf/2204.08458) [[code](https://github.com/hammoudiproject/SuperpixelGridMasks)]
* [Superpixel-Mix -- BMVC 2021] [Robust Semantic Segmentation with Superpixel-Mix](https://arxiv.org/pdf/2108.00968)
* [ICC -- ICPS 2021] [Intra-Class Cutmix for Unbalanced Data Augmentation](https://dl.acm.org/doi/pdf/10.1145/3457682.3457719)
* [ClassMix -- WACV 2021] [ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning](https://arxiv.org/pdf/2007.07936.pdf) [[code](https://github.com/WilhelmT/ClassMix)]

## Others
* [Cutout -- Arxiv 2017] [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)
* [StackMix -- UAI 2021] [StackMix: A complementary Mix algorithm](https://proceedings.mlr.press/v180/chen22b/chen22b.pdf)
* [RandomMix -- Arxiv 2022 ] [RandomMix: A mixed sample data augmentation method with multiple mixed modes](https://arxiv.org/pdf/2205.08728)(2022)
* [HMix \& GMix -- Arxiv 2022] [A Unified Analysis of Mixed Sample Data Augmentation: A Loss Function Perspective](https://arxiv.org/pdf/2208.09913) [[code](https://github.com/naver-ai/hmix-gmix)]
* [DMixup \& DCutmix -- Arxiv 2021] [Observations on K-image Expansion of Image-Mixing Augmentation for Classification](https://arxiv.org/pdf/2110.04248)

# Applications
## Semi-Supervised
* [ICT -- IJCAI 2019] [Interpolation Consistency Training for Semi-Supervised Learning](https://arxiv.org/pdf/1903.03825.pdf?ref=https://githubhelp.com) [[code](https://github.com/vikasverma1077/ICT)]
* [MixMatch -- NIPS 2019] [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://proceedings.neurips.cc/paper/2019/file/1cd138d0499a68f4bb72bee04bbec2d7-Paper.pdf) [[code](https://github.com/google-research/mixmatch)]
* [ReMixMatch -- Arxiv 2019] [ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/pdf/1911.09785.pdf) [[code](https://github.com/google-research/remixmatch)]
* [DivideMix -- ICLR 2020] [DivideMix: Learning with Noisy Labels as Semi-supervised Learning](https://arxiv.org/pdf/2002.07394.pdf) [[code](https://github.com/LiJunnan1992/DivideMix)]
* [CowMix/CowOut -- Arxiv 2020] [Milking CowMask for Semi-Supervised Image Classification](https://arxiv.org/pdf/2003.12022.pdf) [[code](https://github.com/google-research/google-research/tree/master/milking_cowmask)]
* [MixPUL -- Arxiv 2020] [MixPUL: Consistency-based Augmentation for Positive and Unlabeled Learning](https://arxiv.org/pdf/2004.09388.pdf) [[code](https://github.com/Stomach-ache/MixPUL)]
* [P<sup>3</sup>Mix -- ICLR 2022] [Who us your right Mixup partner in positive and unlabeled learning](https://openreview.net/pdf?id=NH29920YEmj)

## Generative Model
* [AAE -- ICPR 2018] [Data Augmentation via Latent Space Interpolation for Image Classification](https://ieeexplore.ieee.org/iel7/8527858/8545020/08545506.pdf?casa_token=pjiWDeNyO-UAAAAA:xzH6WaFN4ik6otSGYZQ4rYsgtGVtyPK9wTyBgT12Ubrfravtdj3mYY-eIHqcXvKWuyi9JQ_Rx14)
* [ACAI -- ICLR 2019] [Understanding and Improving Interpolation in Autoencoders via an Adversarial Regularizer](https://arxiv.org/pdf/1807.07543.pdf) [[code](https://github.com/brain-research/acai)]
* [AMR -- NIPS 2019] [On Adversarial Mixup Resynthesis](https://proceedings.neurips.cc/paper/2019/file/f708f064faaf32a43e4d3c784e6af9ea-Paper.pdf) [[code](https://github.com/christopher-beckham/amr)]

## Domain Adaption
* [VMT -- Arxiv 2019] [Virtual Mixup Training for Unsupervised Domain Adaptation](https://arxiv.org/pdf/1905.04215.pdf) [[code](https://github.com/xudonmao/VMT)]
* [IIMT -- Arxiv 2020] [Improve Unsupervised Domain Adaptation with Mixup Training](https://arxiv.org/pdf/2001.00677)
* [DM-ADA -- AAAI 2020] [Adversarial Domain Adaptation with Domain Mixup](https://ojs.aaai.org/index.php/AAAI/article/view/6123/5979)
* [DMRL -- ECCV 2020] [Dual Mixup Regularized Learning for Adversarial Domain Adaptation](https://arxiv.org/pdf/2007.03141) [[code](https://github.com/YuanWu3/Dual-Mixup-Regularized-Learning-for-Adversarial-Domain-Adaptation)]
* [SLM -- NIPS 2021] [Select, Label, and Mix: Learning Discriminative Invariant Feature Representations for Partial Domain Adaptation](https://arxiv.org/pdf/2012.03358) [[code](https://github.com/CVIR/Select-Label-Mix-SLM-PDA)]

## Metric Learning
1. [Embedding Expansion -- CVPR 2020] [Embedding expansion: Augmentation in embedding space for deep metric learning](http://openaccess.thecvf.com/content_CVPR_2020/papers/Ko_Embedding_Expansion_Augmentation_in_Embedding_Space_for_Deep_Metric_Learning_CVPR_2020_paper.pdf)(2020) [[code](https://github.com/clovaai/embedding-expansion)]
2. [Metrix -- ICLR 2022]: [It Takes Two to Tango: Mixup for Deep Metric Learning](https://arxiv.org/pdf/2106.04990.pdf)(2022) [[code](https://tinyurl.com/metrix-iclr)]

## Contrastive Learning
* [MixCo -- Arxiv 2022] [MixCo: Mix-up Contrastive Learning for Visual Representation](https://arxiv.org/pdf/2010.06300) [[cdoe](https://github.com/Lee-Gihun/MixCo-Mixup-Contrast)]
* [MCL -- PRL 2022] [Mixing up contrastive learning: Self-supervised representation learning for time series](https://www.sciencedirect.com/science/article/pii/S0167865522000502) [[code](https://github.com/Wickstrom/MixupContrastiveLearning)]
* [MixSiam -- Arxiv 2021] [MixSiam: A Mixture-based Approach to Self-supervised Representation Learning](https://arxiv.org/pdf/2111.02679)
* [Un-Mix -- AAAI 2022] [Un-mix: Rethinking Image Mixtures for Unsupervised Visual Representation Learning](https://ojs.aaai.org/index.php/AAAI/article/view/20119/19878) [[code](https://github.com/szq0214/Un-Mix)]
* [Core-tuning -- NIPS 2021] [Unleashing the Power of Contrastive Self-Supervised Visual Models via Contrast-Regularized Fine-Tuning](https://proceedings.neurips.cc/paper/2021/file/fa14d4fe2f19414de3ebd9f63d5c0169-Paper.pdf) [[code](https://github.com/Vanint/Core-tuning)]
* [Feature Transformation -- ICCV 2021] [Improving Contrastive Learning by Visualizing Feature Transformation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhu_Improving_Contrastive_Learning_by_Visualizing_Feature_Transformation_ICCV_2021_paper.pdf) [[code](https://github.com/DTennant/CL-Visualizing-Feature-Transformation)]
* [Mochi -- NIPS 2020] [Hard Negative Mixing for Contrastive Learning](https://proceedings.neurips.cc/paper/2020/file/f7cade80b7cc92b991cf4d2806d6bd78-Paper.pdf) [[code](https://europe.naverlabs.com/mochi)]
* [ProGCL -- ICML] [ProGCL: Rethinking Hard Negative Mining in Graph Contrastive Learning](https://proceedings.mlr.press/v162/xia22b/xia22b.pdf) [[code](https://github.com/junxia97/ProGCL)]
* [BSIM -- Arxiv 2020] [Beyond Single Instance Multi-view Unsupervised Representation Learning](https://arxiv.org/pdf/2011.13356)
* [i-Mix -- ICLR 2021] [i-mix: A domain-agnostic strategy for contrastive representation learning](https://arxiv.org/pdf/2010.08887) [[code](https://github.com/kibok90/imix)]

### NLP
1.  WordMixup & SenMixup: [Augmenting Data with Mixup for Sentence Classification: An Empirical Study](https://arxiv.org/pdf/1905.08941.pdf) [2019; NLP]
2.  TMix & MixText: [MixText: Linguistically-Informed Interpolation of Hidden Space for Semi-Supervised Text Classification](https://arxiv.org/pdf/2004.12239.pdf) [2020; NLP]
3.  Mixup-Transformer: [Mixup-Transformer: Dynamic Data Augmentation for NLP Tasks](https://arxiv.org/pdf/2010.02394) [2020; NLP]
4.  SeqMix: [SeqMix: Augmenting Active Sequence Labeling via Sequence Mixup](https://arxiv.org/pdf/2010.02322) [2020; NLP]
7.  SeqMix: [Sequence-Level Mixed Sample Data Augmentation](https://arxiv.org/pdf/2011.09039)(2020)
8.  AMDA: [Better Robustness by More Coverage: Adversarial and Mixup Data Augmentation for Robust Finetuning](https://aclanthology.org/2021.findings-acl.137.pdf)(2021)
9.  AdvAug: [AdvAug: Robust Adversarial Augmentation for Neural Machine Translation](https://arxiv.org/pdf/2006.11834)(2020)
10. Snippext: [Snippext: Semi-supervised Opinion Mining with Augmented Data](https://dl.acm.org/doi/pdf/10.1145/3366423.3380144)(2020)
11. Emix: [Augmenting NLP models using Latent Feature Interpolations](https://aclanthology.org/2020.coling-main.611)(2020)
12. Calibrated-BERT-Fine-Tuning: [Calibrated Language Model Fine-Tuning for In- and Out-of-Distribution Data](http://arxiv.org/pdf/2010.11506)(2020)
13. STEMM: [STEMM: Self-learning with Speech-text Manifold Mixup for Speech Translation](https://arxiv.org/pdf/2203.10426)(2022)
14. X-Mixup: [Enhancing Cross-lingual Transfer by Manifold Mixup](https://arxiv.org/pdf/2205.04182)(2022)
15. mXEncDec: [Multilingual Mix: Example Interpolation Improves Multilingual Neural Machine Translation](https://arxiv.org/pdf/2203.07627)(2022)


### Unsupervised: 
3.  LSI: [Data Augmentation via Latent Space Interpolation for Image Classification](https://ieeexplore.ieee.org/iel7/8527858/8545020/08545506.pdf)(2018)
### Domain Adaptation 
### Robustness
1.  M-TLAT: [Addressing Neural Network Robustness with Mixup and Targeted Labeling Adversarial Training](https://arxiv.org/pdf/2008.08384) [2022; Robustness]
2. IAT: [Interpolated adversarial training: Achieving robust neural networks without sacrificing too much accuracy](https://dl.acm.org/doi/pdf/10.1145/3338501.3357369)
3. Mixup Inference: [Mixup Inference: Better Exploiting Mixup to Defend Adversarial Attacks](https://arxiv.org/pdf/1909.11515)(2020; CV, Robustness)
4. MixACM: [MixACM: Mixup-Based Robustness Transfer via Distillation of Activated Channel Maps](https://proceedings.neurips.cc/paper/2021/file/240c945bb72980130446fc2b40fbb8e0-Paper.pdf)(2021)
5. AVmixup: [Adversarial Vertex Mixup: Toward Better Adversarially Robust Generalization](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_Adversarial_Vertex_Mixup_Toward_Better_Adversarially_Robust_Generalization_CVPR_2020_paper.pdf)(2020)
6. AOM: [Adversarially Optimized Mixup for Robust Classification](https://arxiv.org/pdf/2103.11589)(2021)
7. VarMixup: [VarMixup: Exploiting the Latent Space for Robust Training and Inference](https://arxiv.org/pdf/2003.06566v1.pdf)(2020)
8. GIF: [Guided Interpolation for Adversarial Training](https://arxiv.org/pdf/2102.07327)
9. Mixup-SSAT: [Semi-supervised Semantics-guided Adversarial Training for Trajectory Prediction](https://arxiv.org/pdf/2205.14230)(2022)
10. MixDiversity: [Mixup Decoding for Diverse Machine Translation](https://arxiv.org/pdf/2109.03402)(2021)

### Federated Learning
1. Mix2FLD: [Mix2FLD: Downlink Federated Learning After Uplink Federated Distillation With Two-Way Mixup](https://ieeexplore.ieee.org/iel7/4234/5534602/09121290.pdf)(2020)
2. XOR Mixup: [XOR Mixup: Privacy-Preserving Data Augmentation for One-Shot Federated Learning](https://arxiv.org/pdf/2006.05148)(2020)
3. FedMix: [FedMix: Approximation of Mixup under Mean Augmented Federated Learning](http://arxiv.org/pdf/2107.00233)(2021)
### Graph
1. GraphMix: [GraphMix: Improved Training of GNNs for Semi-Supervised Learning](https://ojs.aaai.org/index.php/AAAI/article/view/17203/17010)(2021)
### Point Cloud
1. PointMixup: [PointMixup: Augmentation for Point Clouds](https://arxiv.org/pdf/2008.06374)(2020)
2. RSMix: [Regularization Strategy for Point Cloud via Rigidly Mixed Sample](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Regularization_Strategy_for_Point_Cloud_via_Rigidly_Mixed_Sample_CVPR_2021_paper.pdf)(2021)
### Super-resolution
1. CutBlur: [Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy](http://openaccess.thecvf.com/content_CVPR_2020/papers/Yoo_Rethinking_Data_Augmentation_for_Image_Super-resolution_A_Comprehensive_Analysis_and_CVPR_2020_paper.pdf)

### Others
1. C2L: [Comparing to Learn: Surpassing ImageNet Pretraining on Radiographs By Comparing Image Representations](https://arxiv.org/pdf/2007.07423)
2. Mixup: [Mixup Learning Strategies for Text-independent Speaker Verification](https://cse.hkust.edu.hk/~mak/PDF/is2019-mixup.pdf)(2019)
3. Mixup: [An investigation of mixup training strategies for acoustic models in ASR](https://www.isca-speech.org/archive_v0/Interspeech_2018/pdfs/2191.pdf)(2018)
4. VLMixer: [VLMixer: Unpaired Vision-Language Pre-training via Cross-Modal CutMix](https://proceedings.mlr.press/v162/wang22h/wang22h.pdf)(2022)
5. VideoMix: [VideoMix: Rethinking Data Augmentation for Video Classification](https://arxiv.org/pdf/2012.03457)(2020)

# Code Repositories
{_Method_}\* denotes the code of _Method_ is reimplemented by others.
16.  TMix & MixText: https://github.com/GT-SALT/MixText
18.  SeqMix: https://github.com/rz-zhang/SeqMix
24.  Mixup Inference: https://github.com/P2333/Mixup-Inference
25.  FedMix: https://github.com/smduan
27.  GraphMix: https://github.com/vikasverma1077/GraphMix
28.  PointMixup: https://github.com/yunlu-chen/PointMixup
29.  PA-AUG: https://github.com/sky77764/pa-aug.pytorch
30.  RSMix: https://github.com/dogyoonlee/RSMix
32.  Cutblur: https://github.com/clovaai/cutblur
43.  C2L: https://github.com/funnyzhou/C2L_MICCAI2020
50.  MixACM: https://awaisrauf.github.io/MixACM
51.  AVmixup: https://github.com/xuyinhu/AVmixup
54.  AutoMix: https://github.com/Westlake-AI/openmixup
62.  AMDA: https://github.com/thunlp/MixADA
65.  Snippext: https://github.com/rit-git/Snippext_public
68.  Mixup-with-AUM-and-SM: https://github.com/seoyeon-p/MixUp-Guided-by-AUM-and-Saliency-Map
69.  Calibrated-BERT-Fine-Tuning: https://github.com/Lingkai-Kong/Calibrated-BERT-Fine-Tuning
71.  STEMM: https://github.com/ictnlp/STEMM
72.  X-Mixup: https://github.com/yhy1117/X-Mixup
76.  VLMixer: https://github.com/ttengwang/VLMixer
77.  CSANMT: https://github.com/pemywei/csanmt

## Keys
1.  Number of samples for mixup == 2;
2.  Data Augment v.s. Regularization --> Mixup v.s. Weight Decay;
3.  Why and which specific properties;
4.  Limits;
    in some way distorted, i.e., vicinal distribution not match the true distribution of the data
5.  Order of data augmentation;
6.  the role of \alpha;

{_code_}\* denotes the code of _code_ is reimplemented by others.

## Variants
13. Mixup Inference: [Mixup Inference: Better Exploiting Mixup to Defend Adversarial Attacks](https://arxiv.org/pdf/1909.11515)(2020; CV, Robustness)

## Others
3.  MixFeat: [Mixfeat: Mix feature in latent space learns discriminative space](https://openreview.net/pdf?id=HygT9oRqFX) [2019; CV]