# Papers

## Bese Method
1.  Mixup: [Mixup: Beyond Empirical Risk Minimization](https://arxiv.org/pdf/1710.09412) [2018; CV, Speech, Tabular, GAN, Adversarial Example]
2.  Framework: [Improved Mixed-Example Data Augmentation](https://arxiv.org/pdf/1805.11272.pdf?ref=https://githubhelp.com) [2018; CV]

## Variants
1.  CutMix: [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf) [2019; CV, Adversarial Example, Uncertainty]
2.  RICAP: [RICAP: Random Image Cropping and Patching Data Augmentation for Deep CNNs](http://proceedings.mlr.press/v95/takahashi18a/takahashi18a.pdf) [2018, CV]
3.  BC: [Learning from Between-Class Examples for Deep Sound Recognition](https://arxiv.org/pdf/1711.10282.pdf) [2018, Sound]


## Others
1.  SamplePairing: [Data Augmentation by Pairing Samples for Images Classification](https://arxiv.org/pdf/1801.02929) [2018; CV]

## Applications
### CV
1.  BC+: [Between-class Learning for Image Classification](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tokozume_Between-Class_Learning_for_CVPR_2018_paper.pdf) (2018; CV)
### NLP
2.  wordMixup & senMixup: [Augmenting Data with Mixup for Sentence Classification: An Empirical Study](https://arxiv.org/pdf/1905.08941.pdf) [2019; NLP]


# Code Repositories
{_Method_}\* denotes the code of _Method_ is reimplemented by others.
1. Mixip: https://github.com/facebookresearch/mixup-cifar10
2. Framework: https://github.com/ceciliaresearch/MixedExample
3. CutMix: https://github.com/clovaai/CutMix-PyTorch
4. RICAP*: https://github.com/4uiiurz1/pytorch-ricap
5. BC: https://github.com/mil-tokyo/bc_learning_sound
6. BC+: https://github.com/mil-tokyo/bc_learning_image

## Keys
1.  Number of samples for mixup == 2;
2.  Data Augment v.s. Regularization --> Mixup v.s. Weight Decay;
3.  Why and which specific properties;
4.  Limits;
5.  Order of data augmentation;