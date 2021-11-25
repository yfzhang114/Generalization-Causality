* [Generalization/OOD](#generalizationood)
   * [2021](#2021)
   * [2020](#2020)
   * [OLD but Important](#old-but-important)
   * [Survey](#survey)
* [Robutness/Adaptation/Fairness](#robutnessadaptationfairness)
   * [2021](#2021-1)
   * [Before 2021](#before-2021)
* [Causality](#causality)
   * [Individual Treatment Estimation](#individual-treatment-estimation)
* [Data-Centric/Prompt](#data-centricprompt)
   * [Data Centric](#data-centric)
   * [Prompts](#prompts)
* [Optimization/GNN/Energy/Others](#optimizationgnnenergygenerativeothers)
   * [Optimization](#optimization)
   * [LTH (Lottery Ticket Hypothesis)](#lth-lottery-ticket-hypothesis)
   * [Generative Model (Mainly Diffusion Model)](#generative-model-mainly-diffusion-model)
   * [Survey](#survey-1)

Domain generalization, OOD, optimization, data-centric, prompt, robutness以及causality相关问题的前沿文章阅读清单，链接为笔记，没有链接就是还没看.

# Generalization/OOD
## 2021
1. ICCV [CrossNorm and SelfNorm for Generalization under Distribution Shifts](https://zhuanlan.zhihu.com/p/426728622)(思路简单的正则化技术用于DG)
2. ICCV [A Style and Semantic Memory Mechanism for Domain Generalization](https://zhuanlan.zhihu.com/p/426728622)(尝试着去使用intra-domain style invariance来提升模型的泛化性能)
3. Arxiv: [Towards a Theoretical Framework of Out-of-Distribution Generalization](https://zhuanlan.zhihu.com/p/382608823) （新理论）
4. Arxiv(**Yoshua Bengio**) _Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization_ (当OOD遇到信息瓶颈理论)
5. Arxiv _Generalization of Reinforcement Learning with Policy-Aware Adversarial Data Augmentation_
6. Arxiv _Embracing the Dark Knowledge: Domain Generalization Using Regularized Knowledge Distillation_(使用知识蒸馏作为正则化手段)
7. Arxiv _Delving Deep into the Generalization of Vision Transformers under Distribution Shifts_ (视觉transformer的泛化性讨论)
8. Arxiv _Training Data Subset Selection for Regression with Controlled Generalization Error_ (从大量训练实例中选择数据子集，并保持可比的泛化性)
9. Arxiv(**MIT**) _Measuring Generalization with Optimal Transport_ (网络复杂度与泛化性的理论研究，)
10. Arxiv(**SJTU**) [OoD-Bench: Benchmarking and Understanding Out-of-Distribution Generalization Datasets and Algorithms](https://view.inews.qq.com/a/20210615A04V1C00?tbkt=B1&uid=) (揭示OOD的评测标准尚不完善并提出评测方案)
11. Arxiv (Tsinghu) _Domain-Irrelevant Representation Learning for Unsupervised Domain Generalization_ (新的task：无监督的DG，源域的数据标签不可以用)
12. ICML Oral： [Can Subnetwork Structure be the Key to Out-of-Distribution Generalization?](https://zhuanlan.zhihu.com/p/382608823) （彩票模型寻找模型中泛化能力更强的小模型）
13. ICML Oral：[Domain Generalization using Causal Matching](https://zhuanlan.zhihu.com/p/382608823) (contrastive loss特征对齐+特征不变性约束)
14. ICML Oral: _Just Train Twice: Improving Group Robustness without Training Group Information_
15. ICML Spotlight: [Environment Inference for Invariant Learning](https://zhuanlan.zhihu.com/p/382608823) (没有域标签如何学习域不变性特征？)
16. ICLR Poster: [Understanding the failure modes of out-of-distribution generalization](https://zhuanlan.zhihu.com/p/382608823) （OOD失败的两种原因）
17. ICLR Poster: [An Empirical Study of Invariant Risk Minimization](https://openreview.net/forum?id=jrA5GAccy_)(对IRM的实验性探索，如可见域的diversity如何影响IRM性能等)
18. ICLR Poster _In Search of Lost Domain Generalization_ (没有model selection的方法不是好方法，如何根据验证集选择模型？)
19. ICLR Poster [Modeling the Second Player in Distributionally Robust Optimization](https://zhuanlan.zhihu.com/p/381176721)(用对抗学习建模DRO的uncertainty set)
20. ICLR Poster [Learning perturbation sets for robust machine learning](https://zhuanlan.zhihu.com/p/391235069)(使用生成模型学习扰动集合)
21. ICLR Spotlight(**Yoshua Bengio**) [Systematic generalisation with group invariant predictions](https://zhuanlan.zhihu.com/p/382608823) (将每个类分成不同的domain(_environment inference_，然后约束每个域的特征尽可能一致从而避免虚假依赖))
22. CVPR Oral: [Reducing Domain Gap by Reducing Style Bias](https://zhuanlan.zhihu.com/p/382608823) (channel-wise 均值作为图像风格，减少CNN对风格的依赖)
23. AISTATS _Linear Regression Games: Convergence Guarantees to Approximate Out-of-Distribution Solutions_
24. AISTATS Oral _Does Invariant Risk Minimization Capture Invariance_(IRM只有在满足特定条件的情况下才能真正捕捉不变形特征)
25. NeurIPS 2021 [Counterfactual Invariance to Spurious Correlations: Why and How to Pass Stress Tests](https://arxiv.org/abs/2106.00545)(本文使用因果工具设计了一个可行的算法，将反事实推理与域泛化（OOD）联系起来，进行有效的“stress test”，比如变化一个句子包含的的gender信息，看最后情感分类会不会改变。)

## 2020
1. Arxiv [I-SPEC: An End-to-End Framework for Learning Transportable, Shift-Stable Models](https://zhuanlan.zhihu.com/p/288980706)(将Domain Adaptation看作是因果图推理问题)
2. Arxiv (**Stanford**)_Distributionally Robust Lossesfor Latent Covariate Mixtures_.
3. NeurIPS [Energy-based Out-of-distribution Detection](https://zhuanlan.zhihu.com/p/343678039)(使用能量模型检测OOD样本)
4. NeurIPS _Fairness without demographics through adversarially reweighted learning_ (利用对抗学习对难样本进行加权，希望加权后的样本使得分类器的损失更大)
5. NeurIPS _Self-training Avoids Using Spurious Features Under Domain Shift_ (使用target domain的无标签数据训练有助于避免使用虚假特征)
6. NeurIPS _What shapes feature representations? Exploring datasets, architectures, and training_(Simplicity Bias，神经网络倾向于拟合“容易”的特征)
7. Arxiv [Invariant Risk Minimization](https://zhuanlan.zhihu.com/p/273209891) (奠基之作，跳出经验风险最小化--不变风险最小化)
8. ICLR Poster [The Risks of Invariant Risk Minimization](https://zhuanlan.zhihu.com/p/273209891) (不变风险最小化的缺陷:域数目过少IRM即失败)
9. ICLR _Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization_(GroupDRO: 拥有强正则的DRO)
10. ICML _An investigation of why overparameterizationexacerbates spurious correlations_(神经网络的过参数化是造成网络使用虚假相关性的重要原因)
11. ICML UDA workshop _Learning Robust Representations with Score Invariant Learning_(非归一化统计模型：用能量学习的方式做OOD)

## OLD but Important
1. ICML 2018 Oral (**Stanford**) _Fairness Without Demographics in Repeated Loss Minimization._
2. ICCV 2017 [CCSA--Unified Deep Supervised Domain Adaptation and Generalization](https://blog.csdn.net/Adupanfei/article/details/85165667) (对比损失对齐源域目标域样本空间)
3. JSTOR (**Peters**)Causal inference by using invariant prediction: identification and confidence intervals.
4. ICML 2015 [Towards a Learning Theory of Cause-Effect Inference](使用kernel mean embedding和分类器进行casual inference                  )
5. IJCAI 2020 (**CMU**) _Causal Discovery from Heterogeneous/Nonstationary Data_

## Survey
1. [Causality 基础概念汇总](https://zhuanlan.zhihu.com/p/269625734)
2. [Domain Adaptation基础概念与相关文章解读](https://zhuanlan.zhihu.com/p/272508224)


# Robutness/Adaptation/Fairness

## 2021
1. ICLR Poster [Learning perturbation sets for robust machine learning](https://zhuanlan.zhihu.com/p/391235069)(使用生成模型学习扰动集合)
2. ICCV [Generalized Source-free Domain Adaptation](https://zhuanlan.zhihu.com/p/404697072)(不使用源域数据，只有源域预训练的模型时如何adaptation并保证source domain的性能)
3. ICCV [Adaptive Adversarial Network for Source-free Domain Adaptation](https://zhuanlan.zhihu.com/p/426728622)(在模型优化过程中，我们能否寻找一种新的针对目标的分类器，并使其适应目标特征)
4. ICCV [Gradient Distribution Alignment Certificates Better Adversarial Domain Adaptation](https://zhuanlan.zhihu.com/p/426728622)(该算法通过特征提取器和鉴别器之间的对抗学习来减小特征梯度在两个域之间的分布差异)
5. FAccT [Algorithmic recourse: from counterfactual explanations to interventions](https://zhuanlan.zhihu.com/p/424631782)(提出了causal recourse的概念)
6. ICML WorkShop [On the Fairness of Causal Algorithmic Recourse](https://zhuanlan.zhihu.com/p/424631782)(本文在group recourse的基础上考虑了多个变量之间的相互影响即所谓的因果关系。)

## Before 2021
1. Available at Optimization Online [Kullback-Leibler Divergence Constrained Distributionally Robust Optimization](https://zhuanlan.zhihu.com/p/381176721)(开篇之作，使用KL散度构造DRO中的uncertainty set)
2. ICLR 2018 Oral [Certifying Some Distributional Robustnesswith Principled Adversarial Training](https://zhuanlan.zhihu.com/p/381176721)(基于 Wasserstein-ball构造uncertainty set，用于adversarial robustness)
3. ICML 2018 Oral [Does Distributionally Robust Supervised Learning Give Robust Classifiers?](https://zhuanlan.zhihu.com/p/381176721)(DRO就一定比ERM好？不一定！必须引入额外信息)
4. NeurIPS 2019 [Distributionally Robust Optimization and Generalization in Kernel Methods](https://zhuanlan.zhihu.com/p/381176721)(本文使用MMD(maximummean discrepancy)对uncertainty set进行建模，得到了MMD DRO)
5. EMNLP 2019 [Distributionally Robust Language Modeling](https://zhuanlan.zhihu.com/p/381176721)(Coarse-grained mixture models在NLP中的经典案例)
6. Arxiv 2019 [Equalizing recourse across groups](https://zhuanlan.zhihu.com/p/424631782)(基础的recourse测量的是单个样本，本文给出了一个group级别的recourse度量。)

# Causality

## Individual Treatment Estimation
1. ICML 2017 [Estimating individual treatment effect: generalization bounds and algorithms](https://zhuanlan.zhihu.com/p/426793887)(本文第一次提出了ITE的概念，并使用DA的一套理论对其进行bound，依次设计了一套行而有效的算法。)
2. NeurIPS 2019 [Adapting Neural Networks for the Estimation of Treatment Effects](https://zhuanlan.zhihu.com/p/426793887)(这篇文章的核心思想是这样的：我们没必要使用所有的协方差变量X进行adjustment。)
3. PNAS 2019 [Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning](https://zhuanlan.zhihu.com/p/426793887)(本文提出了一种新的框架X-learner，当各个treatment组的数据非常不均衡的时候，这种框架非常有效。)
4. AAAI 2020 [Learning Counterfactual Representations for Estimating Individual Dose-Response Curves](https://zhuanlan.zhihu.com/p/426793887)(本文提出了新的metric，新的数据集，和训练策略，允许对任意数量的treatment的outcome进行估计。)
5. ICLR 2021 Oral: [VCNet and Functional Targeted Regularization For Learning Causal Effects of Continuous Treatments](https://zhuanlan.zhihu.com/p/426793887)(本文基于varying coefficient model，让每个treatment对应的branch成为treatment的函数，而不需要单独设计branch，依次达到真正的连续性。)
6. Arxiv 2021 [Neural Counterfactual Representation Learning for Combinations of Treatments](https://zhuanlan.zhihu.com/p/426793887)(本文考虑更复杂的情况：多种treatment共同作用。)
7. NeurIPS 2021 Spotlight [On Inductive Biases for Heterogeneous Treatment Effect Estimation](https://arxiv.org/abs/2106.03765)(本文提出了新框架FlexTENet，直接对条件因果值τ进行估计，而不是对μ1，μ2分别估计)
8. NeurIPS 2021 [Nonparametric Estimation of Heterogeneous Treatment Effects: From Theory to Learning Algorithms](https://arxiv.org/abs/2101.10943)(本文分析了进来进行 individual treatment effect的各种算法范式，)
9. Arxiv 2021 [Cycle-Balanced Representation Learning For Counterfactual Inference](对treatment，control两个group分别encode，然后对抗学习减少域差距，为了防止分类信息被抹去，加上cycle-consistance的约束，重构特征。)


# Data-Centric/Prompt

## Data Centric
1. AISTATS 2019 [Towards Optimal Transport with Global Invariances](https://zhuanlan.zhihu.com/p/413791971)(如何对齐两个数据集？)
2. NeurIPS 2020 [Geometric Dataset Distances via Optimal Transport](https://zhuanlan.zhihu.com/p/413791971)(如何定义两个数据集之间的距离？)
3. ICML 2021 [Dataset Dynamics via Gradient Flows in Probability Space](https://zhuanlan.zhihu.com/p/413791971)(如何进行数据集优化，使得两个数据集尽可能的像？)

## Prompts

1. ACL 2021 [WARP: Word-level Adversarial ReProgramming](https://zhuanlan.zhihu.com/p/407144573)(Continuous Prompt开篇之作)
2. Arxiv 2021 **Stanford**[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://zhuanlan.zhihu.com/p/407144573)(Continuous Prompt用于NLG的各种任务)(将prompt用于NLG任务上)
3. Arxiv 2021 **Google**[The Power of Scale for Parameter-Efficient Prompt Tuning](https://zhuanlan.zhihu.com/p/407144573)(目前最简单的preifx training：只对input添加prefix)
4. Arxiv 2021 **DeepMind**[Multimodal Few-Shot Learning with Frozen Language Models](https://zhuanlan.zhihu.com/p/407144573)(利用图像编码器把图像作为一种动态的prefix，与文本一起送入LM中)


# Optimization/GNN/Energy/Generative/Others

## Optimization
1. ICML 2021 [An End-to-End Framework for Molecular Conformation Generation via Bilevel Programming](https://zhuanlan.zhihu.com/p/390808626)
2. NeurIPS 2021 _Deep Structural Causal Models for Tractable Counterfactual Inference_
1. ICML 2018 _Bilevel Programming for Hyperparameter Optimization and Meta-Learning_(用bi-level programming建模超参数搜索与meta-learning)
2. NeurIPS 2021 [Energy-based Out-of-distribution Detection](https://zhuanlan.zhihu.com/p/343678039)

## LTH (Lottery Ticket Hypothesis)
1. NeurIPS 2020: [The Lottery Ticket Hypothesis for Pre-trained BERT Networks](https://zhuanlan.zhihu.com/p/404139792) (彩票假设用于BERT fine-tune))
2. ICML 2021 Oral： [Can Subnetwork Structure be the Key to Out-of-Distribution Generalization?](https://zhuanlan.zhihu.com/p/404139792) (彩票假设用于OOD泛化)
3. CVPR 2021: [The Lottery Tickets Hypothesis for Supervised and Self-supervised Pre-training in Computer Vision Models](https://zhuanlan.zhihu.com/p/404139792) (彩票假设用于视觉模型预训练)


## Generative Model (mainly diffusion model)
1. _Estimation of Non-Normalized Statistical Models by Score Matching_(使用分步积分（Score Matching）的方法解决非归一化分布的估计问题)
2. UAI 2019 _Sliced Score Matching: A Scalable Approach to Density and Score Estimation_(将高维的梯度场沿随即方向投影到一维的标量场再进行score-macthing) 
3. NeurIPS 2019 Oral _Generative Modeling by Estimating Gradients of the Data Distribution_(通过添加噪声的方法，增强Langevin MCMC对低概率密度区域的建模能力)
4. NeurIPS 2020 _improved techniques for training score-based generative models_(对score-based generative model失败案例的分析和改进，生成能力开始媲美GAN)
6. NeurIPS 2020 [Denoising Diffusion Probabilistic Models](https://zhuanlan.zhihu.com/p/366004028)(除VAE, GAN, FLOW外又一生成范式)
7. ICLR 2021 **Outstanding Paper Award** [Score-Based Generative Modeling through Stochastic Differential Equations](http://yang-song.github.io/blog/2021/score/)
8. Arxiv 2021 _Diffusion Models Beat GANs on Image Synthesis_(Diffusion Models在图像和合成上超越GAN) 
10. Arxiv 2021 Variational Diffusion Models


## Survey
1. [综述：基于能量的模型](https://zhuanlan.zhihu.com/p/343529491)
