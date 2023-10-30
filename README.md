

This is a repository for organizing articles related to Domain generalization, OOD, optimization, data-centric learning, prompt learning, robutness, and causality. Most papers are linked to **my reading notes**. Feel free to visit my [personal homepage](https://yfzhang114.github.io/) and contact me for collaboration and discussion.

### About Me :high_brightness: 
I'm the second year Ph.D. student at the State Key Laboratory of Pattern Recognition, the University of Chinese Academy of Sciences, advised by Prof. [Tieniu Tan](http://people.ucas.ac.cn/~tantieniu). I have also spent time at Microsoft, advised by Prof. [Jingdong Wang](https://jingdongwang2017.github.io/), alibaba DAMO Academy, work with Prof. [Rong Jin](https://scholar.google.com/citations?user=CS5uNscAAAAJ&hl=zh-CN).


###  🔥 Updated 2023-10-30
- Recent Domain generalization, test time adaptation, and OOD detection papers on **ICCV 2023**, LLM safety have been updated.
- Our paper  [OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling](https://arxiv.org/abs/2309.12659) has been accepted by **NeurIPS 2023**.  [[Code]](https://github.com/yfzhang114/OneNet)[[Reading Notes]](https://zhuanlan.zhihu.com/p/658191974)
- Our paper  [Domain-Specific Risk Minimization for Out-of-Distribution Generalization](https://arxiv.org/abs/2208.08661) has been accepted by **SIGKDD 2023**. [[Code]](https://github.com/yfzhang114/AdaNPC) [[Reading Notes]](https://zhuanlan.zhihu.com/p/631524930)
- Our paper  [AdaNPC: Exploring Non-Parametric Classifier for Test-Time Adaptation](https://arxiv.org/abs/2304.12566) has been accepted by **ICML 2023**. [[Code]](https://github.com/yfzhang114/AdaNPC)  [[Reading Notes]](https://zhuanlan.zhihu.com/p/624770864)
- Our paper [Free Lunch for Domain Adversarial Training: Environment Label Smoothing](https://arxiv.org/abs/2302.00194) has been accepted by **ICLR 2023**. [[Code]](https://github.com/yfzhang114/Environment-Label-Smoothing)  [[Reading Notes]](https://zhuanlan.zhihu.com/p/600466715)
- Our paper [Exploring Transformer Backbones for Heterogeneous Treatment Effect Estimation](https://arxiv.org/abs/2202.01336) has been accepted by **NeurIPS ML Safety** workshop. [[Code]](https://github.com/hlzhang109/TransTEE)
- Our paper Towards Principled Disentanglement for Domain Generalization has been selected for an CVPR **ORAL** presentation. :blush: [[Reading Notes]](https://zhuanlan.zhihu.com/p/477855079) [[Code]](https://github.com/hlzhang109/DDG)  [[paper]](https://arxiv.org/abs/2111.13839)

# Table of Contents (ongoing)
* [Generalization/OOD](#generalizationood)
   * [2023](#2023)
   * [2022](#2022)
   * [2017-2021](#old-but-important)
* [LLM safety](#llm-safety)
* [Test-time adaptation](#test-time-adaptation)
* [Robutness/Adaptation/Fairness/OOD Detection](#robutnessadaptationfairnessood-detection)
   * [2022](#2022-1)
   * [Before 2022](#before-2022)
* [Data-Centric/Prompt/Large-Pretrain-Model](#data-centricpromptlarge-pretrain-model)
   * [Data Centric](#data-centric)
   * [Prompts](#prompts)
   * [Large-Pretrain-Model](#large-pretrain-model)
* [Optimization/GNN/Energy/Causality/Others](#optimizationgnnenergygenerativecausalityothers)
   * [Optimization](#optimization)
   * [Individual Treatment Estimation](#individual-treatment-estimation)
   * [LTH (Lottery Ticket Hypothesis)](#lth-lottery-ticket-hypothesis)
   * [Generative Model (Mainly Diffusion Model)](#generative-model-mainly-diffusion-model)
   * [Implicit Neural Representation (INR)](#implicit-neural-representation-inr)
   * [Survey](#survey)
# Generalization/OOD

## 2023
0. ICLR [Free Lunch for Domain Adversarial Training: Environment Label Smoothing](https://arxiv.org/abs/2302.00194)(环境标签平滑，一行代码提升对抗学习的稳定性和泛化性). [[Code]](https://github.com/yfzhang114/Environment-Label-Smoothing)  [[Reading Notes]](https://zhuanlan.zhihu.com/p/600466715)
1. ICML  [AdaNPC: Exploring Non-Parametric Classifier for Test-Time Adaptation](https://arxiv.org/abs/2304.12566)(用KNN进行测试时间自适应，从理论上分析了TTA work的原因)[[Code]](https://github.com/yfzhang114/AdaNPC)  [[Reading Notes]](https://zhuanlan.zhihu.com/p/624770864)
2. ICLR [Out-of-Distribution Representation Learning for Time Series Classification](https://arxiv.org/abs/2209.07027)(从OOD的角度考虑时序分类的问题)
3. ICLR [Contrastive Learning for Unsupervised Domain Adaptation of Time Series](https://arxiv.org/abs/2206.06243)(用对比学习对其类间分布为时序DA学一个好的表征)
4. ICLR [Pareto Invarian Risk Minimization](https://openreview.net/forum?id=esFxSb_0pSL)(通过多目标优化角度理解与缓解OOD/DG优化难问题)
5. ICLR [Fairness and Accuracy under Domain Generalization](https://arxiv.org/abs/2301.13323)(不仅考虑泛化的性能，也考虑泛化的公平性)
6. Arxiv [Adversarial Style Augmentation for Domain Generalization](https://arxiv.org/abs/2301.12643)(对抗学习添加图像扰动以提升模型泛化性)
7. Arxiv [CLIPood: Generalizing CLIP to Out-of-Distributions](https://arxiv.org/abs/2302.00864)(使用预训练的CLIP模型，克服domain shift and open class两个问题)
9. SIGKDD [Domain-Specific Risk Minimization for Out-of-Distribution Generalization](https://arxiv.org/abs/2208.08661)(每个域学习单独的分类器，测试阶段根据entropy动态组合)[[Code]](https://github.com/yfzhang114/AdaNPC)[[Reading Notes]](https://zhuanlan.zhihu.com/p/631524930)
10. CVPR [Federated Domain Generalization with Generalization Adjustment](https://scholar.google.com/scholar_url?url=https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Federated_Domain_Generalization_With_Generalization_Adjustment_CVPR_2023_paper.pdf&hl=zh-CN&sa=X&d=13348506996942284912&ei=sTpvZIjhI9OQ6rQP29uDqAU&scisig=AGlGAw8T1YjQNN8nVv2lI6LPBiGS&oi=scholaralrt&hist=lUnt8X4AAAAJ:7797965790415635509:AGlGAw-zJ0qtstLHlwZtiYmf7uNN&html=&pos=1&folt=rel)(为联邦域泛化(FedDG)提供了一个新的新的减小方差的正则项以鼓励公平性)
11. CVPR [Distribution Shift Inversion for Out-of-Distribution Prediction](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Distribution_Shift_Inversion_for_Out-of-Distribution_Prediction_CVPR_2023_paper.pdf)(TTA方法，将OoD测试样本用仅在源分布上训练的扩散模型向训练分布转移然后再测试)
12. CVPR [SFP: Spurious Feature-targeted Pruning for Out-of-Distribution Generalization](https://arxiv.org/abs/2305.11615)(通过移除那些强烈依赖已识别的虚假特征的网络分支来实现modular risk minimization (MRM))
13. CVPR [Improved Test-Time Adaptation for Domain Generalization](https://arxiv.org/abs/2304.04494)(使用一个具有可学习参数的损失函数，而不是预定义的函数)
14. ICLR [Modeling the Data-Generating Process is Necessary for Out-of-Distribution Generalization](https://openreview.net/forum?id=uyqks-LILZX)(真实世界的数据通常在不同属性上有多种分布偏移，目前DG算法无法work，本文利用数据生成过程的知识自适应地识别和应用正确的正则化约束)
15. ICLR [Using Language to Extend to Unseen Domains](https://openreview.net/forum?id=eR2dG8yjnQ)(利用CLIP模型的知识将源域图像embedding转换为多个目标域的representation（从photos of birds转化为paintings of birds）)
16. ICLR [How robust is unsupervised representation learning to distribution shift?](https://openreview.net/forum?id=LiXDW7CF94J)(无监督学习算法中学习到的表示在各种极端和现实分布变化下的泛化效果优于SL)
17. ICLR [PGrad: Learning Principal Gradients For Domain Generalization](https://openreview.net/forum?id=CgCmwcfgEdH)(测量了所有训练域的训练动态,最终的梯度聚合了并给出一个鲁棒的优化方向，有点像meta-learning)
18. ICLR [Causal Balancing for Domain Generalization](https://openreview.net/forum?id=F91SROvVJ_6)(提出了一种平衡的小批量抽样策略，将有偏差的数据分布转换为平衡分布，基于数据生成过程的潜在因果机制的不变性。)
19. ICLR [Cycle-consistent Masked AutoEncoder for Unsupervised Domain Generalization](https://openreview.net/forum?id=wC98X1qpDBA)(无监督域泛化(UDG)，其中不需要成对的数据来连接不同的域。这个问题的研究相对较少，但在DG背景下是有意义的。)
20. Arxiv [Revisiting Out-of-distribution Robustness in NLP: Benchmark, Analysis, and LLMs Evaluations](https://arxiv.org/pdf/2306.04618.pdf)(泛化算法在NLP benchmark上的表现不比fully finetune好多少特别是ID数据足够多时)
21. Arixv [A Survey on Out-of-Distribution Evaluation of Neural NLP Models](https://arxiv.org/pdf/2306.15261.pdf)(系统评估Adversarial robustness, domain generalization and dataset biases)

## 2022

0. CVPR Oral [Towards Principled Disentanglement for Domain Generalization](https://zhuanlan.zhihu.com/p/477855079)(将解耦用于DG，新理论，新方法)
1. Arxiv [How robust are pre-trained models to distribution shift?](https://arxiv.org/abs/2206.08871)(自监督模型比有监督以及无监督模型更鲁棒，在小部分OOD数据上重新训练classifier提升很大)
2. ICML [A Closer Look at Smoothness in Domain Adversarial Training](https://arxiv.org/abs/2206.08213)(平滑分类损失可以提高域对抗训练的泛化性能)
3. CVPR [Bayesian Invariant Risk Minimization](https://zhuanlan.zhihu.com/p/528829486)(缓解IRM在模型过拟合时退化为ERM的问题)
4. CVPR [Towards Unsupervised Domain Generalization](https://zhuanlan.zhihu.com/p/528829486)(关注模型预训练的过程对DG任务的影响，设计了一个在DG数据集无监督预训练的算法)
5. CVPR [PCL: Proxy-based Contrastive Learning for Domain Generalization](https://zhuanlan.zhihu.com/p/528829486)(直接采用有监督的对比学习用于DG效果并不好，本文提出可行方法)
6. CVPR [Style Neophile: Constantly Seeking Novel Styles for Domain Generalization](https://zhuanlan.zhihu.com/p/528829486)(本文提出了一种新的方法，能够产生更多风格的数据)
7. Arxiv [WOODS: Benchmarks for Out-of-Distribution Generalization in Time Series Tasks](https://woods-benchmarks.github.io/)(一个关于时序数据OOD的多个benchmark)
8. Arxiv [A Broad Study of Pre-training for Domain Generalization and Adaptation](https://arxiv.org/pdf/2203.11819.pdf)(深入研究了预训练对于DA,DG任务的作用，简单的使用目前最好的backbone足已取得SOTA的效果)
9. Arxiv [Domain Generalization by Mutual-Information Regularization with Pre-trained Models](https://arxiv.org/pdf/2203.10789.pdf)(使用预训练模型的特征指导finetune的过程，提高泛化能力)
10. ICLR Oral [A Fine-Grained Analysis on Distribution Shift](https://zhuanlan.zhihu.com/p/466675818)(如何准确的定义distribution shift，以及如何系统的测量模型的鲁棒性)
11. ICLR Oral [Fine-Tuning Distorts Pretrained Features and Underperforms Out-of-Distribution](https://zhuanlan.zhihu.com/p/466675818)(fine-tuning（微调）和linear probing相辅相成)
12. ICLR Spotlight [Towards a Unified View of Parameter-Efficient Transfer Learning](https://zhuanlan.zhihu.com/p/466675818)(统一的参数高效微调理论框架)
13. ICLR Spotlight [How Do Vision Transformers Work?](https://zhuanlan.zhihu.com/p/466675818)(Vision Transformers (ViTs)的优良特性)
14. ICLR Spotlight [On Predicting Generalization using GANs](https://zhuanlan.zhihu.com/p/466675818)(使用源域数据训练出的GAN来预测测试误差)
15. ICLR Poster [Uncertainty Modeling for Out-of-Distribution Generalization](https://zhuanlan.zhihu.com/p/466675818)(域泛化时考虑特征的不确定性，一种新的数据增强方法)
16. ICLR Poster [Gradient Matching for Domain Generalization](https://zhuanlan.zhihu.com/p/466675818)(鼓励来自不同域的梯度之间的内积更大)
17. ICML [DNA: Domain Generalization with Diversified Neural Averaging](https://zhuanlan.zhihu.com/p/553511043)(classifier ensemble，即对分类器进行集成。本文从理论和实验角度讨论了ensemble与DG任务的connection。)
18. ICML [Model Agnostic Sample Reweighting for Out-of-Distribution Learning](https://zhuanlan.zhihu.com/p/553511043)(bi-level的去找一种有效的训练样本加权方式)
19. ICML [Sparse Invariant Risk Minimization](https://zhuanlan.zhihu.com/p/553511043)(利用全局稀疏性约来防止伪特征在训练过程被使用)
20. Arxiv [grounding visual representations with texts for domain generalization](http://arxiv.org/abs/2207.10285)(用跨模态的数据作为模型的监督信息可以提升泛化性)
21. Arxiv [On the Strong Correlation Between Model Invarianceand Generalization](http://arxiv.org/abs/2207.07065)(模型预测的不变性与泛化性有强相关，这里的不变性是对x不同perturbation预测的不变性)
22. NeurIPS [Probable Domain Generalization via Quantile Risk Minimization](https://arxiv.org/abs/2207.09944)(将DG建模成概率泛化的问题，既不是worst-case，也不是average performance)
23. NeurIPS [Improving Multi-Task Generalization via Regularizing Spurious Correlation](https://arxiv.org/abs/2205.09797)(去除对任务标签的虚假依赖，从而提升多任务学习的效果)
24. NeurIPS [Understanding the Generalization Benefit of Normalization Layers: Sharpness Reduction](https://arxiv.org/abs/2206.07085)(从理论上解释归一化层使得损失面锐度降低，GD更易优化)
25. NeurIPS [Assaying Out-Of-Distribution Generalization in Transfer Learning](https://zhuanlan.zhihu.com/p/573246040)(全面的对模型鲁棒性的定义提出新的见解)
26. NeurIPS [On the Strong Correlation Between Model Invariance and Generalization](https://zhuanlan.zhihu.com/p/573246040)(对对泛化与不变性之间的关系进行定量的分析)
27. NeurIPS [Ensemble of Averages: Improving Model Selectionand Boosting Performance in Domain Generalization](https://zhuanlan.zhihu.com/p/573246040)(训练过程中OOD数据性能波动很大)
28. NeurIPS [Diverse Weight Averaging for Out-of-Distribution Generalization](https://zhuanlan.zhihu.com/p/573246040)(沿着训练轨迹平均获得的权重)
29. NeurIPS [Improving Out-of-Distribution Generalization byAdversarial Training with Structured Priors](https://openreview.net/forum?id=Ku1afTnmozi)(使用domain specific structured low-rank perturbations来对抗学习提升OOD性能)
29. NeurIPS Outstanding [On-Demand Sampling:Learning Optimally from Multiple Distributions](https://arxiv.org/pdf/2210.12529.pdf)(一个有理论保证的多域学习算法，达到了目前最低的sample complexity)
30. Arxiv [On Feature Learning in the Presence of Spurious Correlations](http://arxiv.org/abs/2210.11369)(ERM已经能够学到很好的特征了)
31. Arxiv [Simulating Bandit Learning from User Feedback for Extractive Question Answering](https://arxiv.org/abs/2203.10079)(引入少量human evaluation可以提升模型泛化性)
32. ICLR [Uncertainty Modeling for Out-of-Distribution Generalization](https://arxiv.org/abs/2202.03958)(改变图象均值/方差来做数据增强，均值方差考虑batch中的不确定性)

## OLD but Important
****
**2021**

1. ICML [Improved OOD Generalization via Adversarial Training and Pre-training](https://proceedings.mlr.press/v139/yi21a.html)(从理论上表明，一个预先训练的模型对输入扰动具有更强的鲁棒性，那么对下游OOD数据的泛化可以提供更好的初始化。)
2. ICCV [CrossNorm and SelfNorm for Generalization under Distribution Shifts](https://zhuanlan.zhihu.com/p/426728622)(思路简单的正则化技术用于DG)
3. ICCV [A Style and Semantic Memory Mechanism for Domain Generalization](https://zhuanlan.zhihu.com/p/426728622)(尝试着去使用intra-domain style invariance来提升模型的泛化性能)
4. Arxiv: [Towards a Theoretical Framework of Out-of-Distribution Generalization](https://zhuanlan.zhihu.com/p/382608823) （新理论）
5. Arxiv(**Yoshua Bengio**) _Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization_ (当OOD遇到信息瓶颈理论)
6. Arxiv _Generalization of Reinforcement Learning with Policy-Aware Adversarial Data Augmentation_
7. Arxiv _Embracing the Dark Knowledge: Domain Generalization Using Regularized Knowledge Distillation_(使用知识蒸馏作为正则化手段)
8. Arxiv _Delving Deep into the Generalization of Vision Transformers under Distribution Shifts_ (视觉transformer的泛化性讨论)
9. Arxiv _Training Data Subset Selection for Regression with Controlled Generalization Error_ (从大量训练实例中选择数据子集，并保持可比的泛化性)
10. Arxiv(**MIT**) _Measuring Generalization with Optimal Transport_ (网络复杂度与泛化性的理论研究，)
11. Arxiv(**SJTU**) [OoD-Bench: Benchmarking and Understanding Out-of-Distribution Generalization Datasets and Algorithms](https://view.inews.qq.com/a/20210615A04V1C00?tbkt=B1&uid=) (揭示OOD的评测标准尚不完善并提出评测方案)
12. Arxiv (Tsinghu) _Domain-Irrelevant Representation Learning for Unsupervised Domain Generalization_ (新的task：无监督的DG，源域的数据标签不可以用)
13. ICML Oral： [Can Subnetwork Structure be the Key to Out-of-Distribution Generalization?](https://zhuanlan.zhihu.com/p/382608823) （彩票模型寻找模型中泛化能力更强的小模型）
14. ICML Oral：[Domain Generalization using Causal Matching](https://zhuanlan.zhihu.com/p/382608823) (contrastive loss特征对齐+特征不变性约束)
15. ICML Oral: _Just Train Twice: Improving Group Robustness without Training Group Information_
16. ICML Spotlight: [Environment Inference for Invariant Learning](https://zhuanlan.zhihu.com/p/382608823) (没有域标签如何学习域不变性特征？)
17. ICLR Poster: [Understanding the failure modes of out-of-distribution generalization](https://zhuanlan.zhihu.com/p/382608823) （OOD失败的两种原因）
18. ICLR Poster: [An Empirical Study of Invariant Risk Minimization](https://openreview.net/forum?id=jrA5GAccy_)(对IRM的实验性探索，如可见域的diversity如何影响IRM性能等)
19. ICLR Poster _In Search of Lost Domain Generalization_ (没有model selection的方法不是好方法，如何根据验证集选择模型？)
20. ICLR Poster [Modeling the Second Player in Distributionally Robust Optimization](https://zhuanlan.zhihu.com/p/381176721)(用对抗学习建模DRO的uncertainty set)
21. ICLR Poster [Learning perturbation sets for robust machine learning](https://zhuanlan.zhihu.com/p/391235069)(使用生成模型学习扰动集合)
22. ICLR Spotlight(**Yoshua Bengio**) [Systematic generalisation with group invariant predictions](https://zhuanlan.zhihu.com/p/382608823) (将每个类分成不同的domain(_environment inference_，然后约束每个域的特征尽可能一致从而避免虚假依赖))
23. CVPR Oral: [Reducing Domain Gap by Reducing Style Bias](https://zhuanlan.zhihu.com/p/382608823) (channel-wise 均值作为图像风格，减少CNN对风格的依赖)
24. AISTATS _Linear Regression Games: Convergence Guarantees to Approximate Out-of-Distribution Solutions_
25. AISTATS Oral _Does Invariant Risk Minimization Capture Invariance_(IRM只有在满足特定条件的情况下才能真正捕捉不变形特征)
26. NeurIPS [Counterfactual Invariance to Spurious Correlations: Why and How to Pass Stress Tests](https://arxiv.org/abs/2106.00545)(本文使用因果工具设计了一个可行的算法，将反事实推理与域泛化（OOD）联系起来，进行有效的“stress test”，比如变化一个句子包含的的gender信息，看最后情感分类会不会改变。)
27. NeurIPS [Adaptive Risk Minimization: Learning to Adapt to Domain Shift](https://zhuanlan.zhihu.com/p/357962431)(利用未标记的数据来更好地处理新domain引起的distribution shift)
28. NeurIPS [An Empirical Investigation of Domain Generalization with Empirical Risk Minimizers](https://zhuanlan.zhihu.com/p/357962431)(基于domain adaptation的理论测量方法不能准确地捕捉OOD泛化行为)
29. NeurIPS Spotlight [Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://zhuanlan.zhihu.com/p/357962431)(在test的阶段，我们在依然会选择更新模型头部的linear层)
30. NeurIPS [Why Do Better Loss Functions Lead to Less Transferable Features?](https://zhuanlan.zhihu.com/p/357962431)(本文研究了训练目标的选择如何影响卷积神经网络在ImageNet上训练得到的可迁移性)、
****

1. Arxiv 2020 [I-SPEC: An End-to-End Framework for Learning Transportable, Shift-Stable Models](https://zhuanlan.zhihu.com/p/288980706)(将Domain Adaptation看作是因果图推理问题)
2. Arxiv 2020 (**Stanford**)_Distributionally Robust Lossesfor Latent Covariate Mixtures_.
3. NeurIPS 2020 [Energy-based Out-of-distribution Detection](https://zhuanlan.zhihu.com/p/343678039)(使用能量模型检测OOD样本)
4. NeurIPS 2020 _Fairness without demographics through adversarially reweighted learning_ (利用对抗学习对难样本进行加权，希望加权后的样本使得分类器的损失更大)
5. NeurIPS 2020 _Self-training Avoids Using Spurious Features Under Domain Shift_ (使用target domain的无标签数据训练有助于避免使用虚假特征)
6. NeurIPS 2020 _What shapes feature representations? Exploring datasets, architectures, and training_(Simplicity Bias，神经网络倾向于拟合“容易”的特征)
7. Arxiv 2020 [Invariant Risk Minimization](https://zhuanlan.zhihu.com/p/273209891) (奠基之作，跳出经验风险最小化--不变风险最小化)
8. ICLR 2020 Poster [The Risks of Invariant Risk Minimization](https://zhuanlan.zhihu.com/p/273209891) (不变风险最小化的缺陷:域数目过少IRM即失败)
9. ICLR 2020 _Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization_(GroupDRO: 拥有强正则的DRO)
10. ICML 2020 _An investigation of why overparameterizationexacerbates spurious correlations_(神经网络的过参数化是造成网络使用虚假相关性的重要原因)
11. ICML 2020 UDA workshop _Learning Robust Representations with Score Invariant Learning_(非归一化统计模型：用能量学习的方式做OOD)
12. ICML 2018 Oral (**Stanford**) _Fairness Without Demographics in Repeated Loss Minimization._
13. ICCV 2017 [CCSA--Unified Deep Supervised Domain Adaptation and Generalization](https://blog.csdn.net/Adupanfei/article/details/85165667) (对比损失对齐源域目标域样本空间)
14. JSTOR (**Peters**)Causal inference by using invariant prediction: identification and confidence intervals.
15. ICML 2015 [Towards a Learning Theory of Cause-Effect Inference](使用kernel mean embedding和分类器进行casual inference                  )
16. IJCAI 2020 (**CMU**) _Causal Discovery from Heterogeneous/Nonstationary Data_

****
**Survey**

1. [Causality 基础概念汇总](https://zhuanlan.zhihu.com/p/269625734)
2. [Domain Adaptation基础概念与相关文章解读](https://zhuanlan.zhihu.com/p/272508224)
****

# LLM Safety

1. [Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned](https://zhuanlan.zhihu.com/p/664096097)(作者描述了他们早期进行手动红队测试的努力，旨在提高模型的安全性并测量模型的安全性)
2. [Jailbroken: How Does LLM Safety Training Fail?](https://zhuanlan.zhihu.com/p/664096097)(调查为什么这些越狱攻击成功以及它们如何生成的。竞争目标和不匹配的泛化)
3. [Constitutional AI: Harmlessness from AI Feedback](https://zhuanlan.zhihu.com/p/664096097)(通过AI指导来生开发一个有帮助、诚实、无害且不会规避问题的AI助手)
4. [Generative Judge for Evaluating Alignment](https://zhuanlan.zhihu.com/p/664096097)(AUTO-J，相比于传统的评估score，这是一个开源模型，能够有效地评估LLMs在各种任务上的表现。)
5. [Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation](https://zhuanlan.zhihu.com/p/664096097)(通过操纵decoding方法的变化来破坏模型的对齐。)
6. [AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models](https://zhuanlan.zhihu.com/p/664096097)(AutoDAN可以通过精心设计的分层遗传算法自动生成隐蔽越狱提示。)
7. [Are aligned neural networks adversarially aligned?](https://zhuanlan.zhihu.com/p/664096097)( 对多模态模型，非NLP的方法进行越狱效果是很不错的,这种错位攻击是十分危险的, 并且目前的对齐技术无法防范此类攻击.)
8. [On the Exploitability of Instruction Tuning](https://zhuanlan.zhihu.com/p/664096097)(这项研究旨在探究如何通过向训练数据中注入特定的遵循指令示例，从而有意改变模型行为的方式)

# Test-time Adaptation

0. ICML  [AdaNPC: Exploring Non-Parametric Classifier for Test-Time Adaptation](https://arxiv.org/abs/2304.12566)(用KNN进行测试时间自适应，从理论上分析了TTA work的原因)[[Code]](https://github.com/yfzhang114/AdaNPC)  [[Reading Notes]](https://zhuanlan.zhihu.com/p/624770864)
1. NeurIPS 2021 [Spotlight] [Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization](https://zhuanlan.zhihu.com/p/559916666)(在test的阶段，我们在依然会选择更新模型头部的linear层)
2. CVPR 2021 [Adaptive Methods for Real-World Domain Generalization](https://zhuanlan.zhihu.com/p/559916666)(测试时输入source domain embedding，即test时利用domain信息)
3. ICLR 2021 [Spotlight] [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://zhuanlan.zhihu.com/p/559916666)(测试时最小化模型预测的entropy)
4. ICCV 2021 [Test-Agnostic Long-Tailed Recognition y Test-Time Aggregating Diverse Experts with Self-Supervision](https://zhuanlan.zhihu.com/p/559916666)(测试时优化样本的自监督损失)
5. NeurIPS 2022 [Revisiting Realistic Test-Time Training: Sequential Inference and Adaptation by Anchored Clustering](https://arxiv.org/abs/2206.02721)(发现源和目标域中的集群，并将目标集群与源集群进行匹配，以改进泛化。)
6. NeurIPS 2022 [Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models](https://arxiv.org/abs/2209.07511)(测试阶段根据最小化预测熵从而更新prompt)
7. NeurIPS 2022 [MEMO: Test Time Robustness via Adaptation and Augmentation](https://arxiv.org/abs/2110.09506)(测试阶段数据增强+最小化熵)
8. NeurIPS 2022 [Test-Time Adaptation via Conjugate Pseudo-labels](https://arxiv.org/abs/2207.09640)(对于现存的有/无监督的TTA loss，可以通过对偶的推导简单的得到最优TTA loss（从广泛的函数类别中进行元学习以获得的最佳的TTA损失），这个loss可以转化为一类特殊的伪标签，被称为 Conjugate Pseudo-labels)
9. CVPR 2022 [Continual Test-Time Domain Adaptation](https://arxiv.org/abs/2203.13591)(从一个源域adapt到一系列连续改变的目标域)
10. Arxiv [A Simple Test-Time Method for Out-of-Distribution Detection](https://arxiv.org/pdf/2207.08210.pdf)(test time adaptation for OOD detection)
11. SIGKDD 2023 [Domain-Specific Risk Minimization for Out-of-Distribution Generalization](https://arxiv.org/abs/2208.08661)(每个域学习单独的分类器，测试阶段根据entropy动态组合)[[Code]](https://github.com/yfzhang114/AdaNPC)[[Reading Notes]](https://zhuanlan.zhihu.com/p/631524930)
12. CVPR 2023 [Improved Test-Time Adaptation for Domain Generalization](https://arxiv.org/abs/2304.04494)(使用一个具有可学习参数的损失函数，而不是预定义的函数)
13. CVPR 2023 [Feature Alignment and Uniformity for Test Time Adaptation](https://arxiv.org/abs/2303.10902)(将TTA作为一个由于源域和目标域之间的域差距而导致的特征修订问题:确保当前批和所有先前批之间的表示之间的均匀性,一致性)
14. CVPR 2023 [TIPI: Test Time Adaptation with Transformation Invariance](https://atuannguyen.com/assets/pdf/nguyen2023tipi.pdf)(为了克服小batch的问题提出了一个新的loss)
15. ICLR 2023 Oral [Towards Stable Test-Time Adaptation in Dynamic Wild World](https://openreview.net/forum?id=g2YraF75Tj)(测试数据流可能具有混合域偏移、小批量和不平衡标签分布shift)
16. ICLR 2023 [Towards Understanding GD with Hard and Conjugate Pseudo-labels for Test-Time Adaptation](https://openreview.net/forum?id=FJXf1FXN8C)(带共轭标签的GD收敛于在高斯模型下任意小误差的最优预测器，而带有传统伪标签的GD在此任务中失败。)
17. ICLR 2023  [Energy-Based Test Sample Adaptation for Domain Generalization](https://openreview.net/forum?id=3dnrKbeVatv)(将目标域数据adapt到源域，利用随机梯度朗之万动力学的能量最小化方法迭代更新样本)
18. ICLR 2023 [Deja Vu: Continual Model Generalization for Unseen Domains ](https://openreview.net/forum?id=L8iZdgeKmI6)(Continual Domain Shift Learning (CDSL))
19. ICLR 2023 [MECTA: Memory-Economic Continual Test-Time Model Adaptation](https://openreview.net/forum?id=N92hjSf5NNh)(目前大多数TTA方法内存消耗比较高因为要反向传播，本文建议减少批处理大小，采用自适应规范化层来保持稳定和准确的预测，并启发式地停止反向传播缓存。另一方面，我们对网络进行修剪以减少优化过程中的计算和内存开销，并在优化后恢复参数以避免遗忘)
20. ICML 2023 Oral [The Price of Differential Privacy under Continual Observation](https://zhuanlan.zhihu.com/p/639174050)(连续适应模型场景下的差分隐私)
21. ICML 2023 Oral [ODS: Test-Time Adaptation in the Presence of Open-World Data Shift](https://zhuanlan.zhihu.com/p/639174050)(同时适应协变量和标签分布的偏移)
22. ICML 2023 [Uncovering Adversarial Risks of Test-Time Adaptation](https://zhuanlan.zhihu.com/p/649295930)(测试批处理中引入恶意样本可能会对最终预测模型的生成产生影响)
23. ICML 2023 [On Pitfalls of Test-Time Adaptation](https://zhuanlan.zhihu.com/p/649295930)(名为TTAB的测试时自适应基准，包含了十种最先进的算法、多样化的分布偏移情况和两种评估协议。)
24. ICML 2023 [Leveraging Proxy of Training Data for Test-Time Adaptation](https://zhuanlan.zhihu.com/p/649295930)(使用训练数据的轻量级且信息丰富的代理方法，并提出了一种完全利用这些代理的测试阶段自适应方法)
25. ICML 2023 [Test-time Adaptation with Slot-Centric Models](https://zhuanlan.zhihu.com/p/649295930)(在场景分解任务中，简单的TTA损失对于任务是不足够的)
26. ICML 2023 [Test-Time Style Shifting: Handling Arbitrary Styles in Domain Generalization](https://zhuanlan.zhihu.com/p/649295930)(将测试样本的样式（与源域存在较大样式差距）转换为模型已熟悉的最近的源域样式，然后进行预测)


# Robutness/Adaptation/Fairness/OOD Detection

1. ICML 2023 Oral [Delving into Noisy Label Detection with Clean Data](https://zhuanlan.zhihu.com/p/639174050)(噪声标签检测中对干净数据的利用，以提高噪声标签检测的性能)

## 2022
1. Arxiv [Are Vision Transformers Robust to Spurious Correlations?](https://arxiv.org/pdf/2203.09125.pdf)(对ViT鲁棒性的研究，更大的模型和更多的训练前数据可以显著提高对伪相关的鲁棒性，预训练数据较少反而不如CNN)
2. CVPR [Exploring Domain-Invariant Parameters for Source FreeDomain Adaptation](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Exploring_Domain-Invariant_Parameters_for_Source_Free_Domain_Adaptation_CVPR_2022_paper.pdf)(相比于以往工作探索域不变特征，该工作旨在寻找域不变参数)
3. CVPR [CENet: Consolidation-and-Exploration Network for Continuous DomainAdaptation](https://openaccess.thecvf.com/content/CVPR2022W/RoSe/papers/Zhang_CENet_Consolidation-and-Exploration_Network_for_Continuous_Domain_Adaptation_CVPRW_2022_paper.pdf)(本文说他提出了continuous DA的概念，但是ICML 18就已经提出了呀？)
4. CVPR [Slimmable Domain Adaptation](https://arxiv.org/abs/2206.06620)(Adaptation的对象不仅应该是数据，本文考虑下游设备的adaptation。)
5. NeurIPS Outstanding [Is Out-of-Distribution Detection Learnable?](https://arxiv.org/pdf/2210.14707.pdf)(各种场景下的OOD detection的PAC理论)
6. ICML [Out-of-Distribution Detection with Deep Nearest Neighbors](https://arxiv.org/pdf/2204.06507.pdf)(用KNN做OOD detection)
7. Arxiv [A Simple Test-Time Method for Out-of-Distribution Detection](https://arxiv.org/pdf/2207.08210.pdf)(test time adaptation for OOD detection)
8. Arxiv [RobArch: Designing Robust Architectures against Adversarial Attacks](https://shengyun-peng.github.io/papers/22_robarch.pdf)(对如何设计鲁棒性更强的模型结构做了大量的实验验证)
9. ICCV [Nearest Neighbor Guidance for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Park_Nearest_Neighbor_Guidance_for_Out-of-Distribution_Detection_ICCV_2023_paper.pdf)(用ID数据给knn-ood再计算一个额外的正则项，缓解knn-ood在near-ID附近表现不好的现象)
10. ICCV [Understanding the Feature Norm for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Park_Understanding_the_Feature_Norm_for_Out-of-Distribution_Detection_ICCV_2023_paper.pdf)( Feature Norm与分类器置信度有关，但是不依赖于类别标签，是一个很好的无监督metric)

## before 2022
1. ICLR Poster [Learning perturbation sets for robust machine learning](https://zhuanlan.zhihu.com/p/391235069)(使用生成模型学习扰动集合)
2. ICCV [Generalized Source-free Domain Adaptation](https://zhuanlan.zhihu.com/p/404697072)(不使用源域数据，只有源域预训练的模型时如何adaptation并保证source domain的性能)
3. ICCV [Adaptive Adversarial Network for Source-free Domain Adaptation](https://zhuanlan.zhihu.com/p/426728622)(在模型优化过程中，我们能否寻找一种新的针对目标的分类器，并使其适应目标特征)
4. ICCV [Gradient Distribution Alignment Certificates Better Adversarial Domain Adaptation](https://zhuanlan.zhihu.com/p/426728622)(该算法通过特征提取器和鉴别器之间的对抗学习来减小特征梯度在两个域之间的分布差异)
5. FAccT [Algorithmic recourse: from counterfactual explanations to interventions](https://zhuanlan.zhihu.com/p/424631782)(提出了causal recourse的概念)
6. ICML WorkShop [On the Fairness of Causal Algorithmic Recourse](https://zhuanlan.zhihu.com/p/424631782)(本文在group recourse的基础上考虑了多个变量之间的相互影响即所谓的因果关系。)
7. NeurIPS [Domain Adaptation with Invariant Representation Learning: What Transformations to Learn?](https://zhuanlan.zhihu.com/p/316265317)(DA为什么需要两个encoder？)
8. NeurIPS [Gradual Domain Adaptation without Indexed Intermediate Domains](https://zhuanlan.zhihu.com/p/316265317)(没有domaparameterin label的Gradual domain adaption(GDA))
9. NeurIPS [Implicit Semantic Response Alignment for Partial Domain Adaptation](https://zhuanlan.zhihu.com/p/316265317)(PDA如何利用好额外类)
10. NeurIPS [The balancing principle for parameter choice in distance-regularized domain adaptation](https://zhuanlan.zhihu.com/p/316265317)(如何挑选分类损失和正则化项的tradeoff parameter)
11. AAAI [Provable Guarantees for Understanding Out-of-distribution Detection](https://arxiv.org/pdf/2112.00787.pdf)(基于数据是高斯混合的假设给出最优density估计方式)
12. Available at Optimization Online [Kullback-Leibler Divergence Constrained Distributionally Robust Optimization](https://zhuanlan.zhihu.com/p/381176721)(开篇之作，使用KL散度构造DRO中的uncertainty set)
13. ICLR 2018 Oral [Certifying Some Distributional Robustnesswith Principled Adversarial Training](https://zhuanlan.zhihu.com/p/381176721)(基于 Wasserstein-ball构造uncertainty set，用于adversarial robustness)
14. ICML 2018 Oral [Does Distributionally Robust Supervised Learning Give Robust Classifiers?](https://zhuanlan.zhihu.com/p/381176721)(DRO就一定比ERM好？不一定！必须引入额外信息)
15. NeurIPS 2019 [Distributionally Robust Optimization and Generalization in Kernel Methods](https://zhuanlan.zhihu.com/p/381176721)(本文使用MMD(maximummean discrepancy)对uncertainty set进行建模，得到了MMD DRO)
16. EMNLP 2019 [Distributionally Robust Language Modeling](https://zhuanlan.zhihu.com/p/381176721)(Coarse-grained mixture models在NLP中的经典案例)
17. Arxiv 2019 [Equalizing recourse across groups](https://zhuanlan.zhihu.com/p/424631782)(基础的recourse测量的是单个样本，本文给出了一个group级别的recourse度量。)
18. ICML 2020 Oral [Continuously indexed domain adaptation](https://zhuanlan.zhihu.com/p/316265317)(连续变化的domain)

# Data-Centric/Prompt/Large-Pretrain-Model

## Data Centric
1. AISTATS 2019 [Towards Optimal Transport with Global Invariances](https://zhuanlan.zhihu.com/p/413791971)(如何对齐两个数据集？)
2. NeurIPS 2020 [Geometric Dataset Distances via Optimal Transport](https://zhuanlan.zhihu.com/p/413791971)(如何定义两个数据集之间的距离？)
3. ICML 2021 [Dataset Dynamics via Gradient Flows in Probability Space](https://zhuanlan.zhihu.com/p/413791971)(如何进行数据集优化，使得两个数据集尽可能的像？)

## Prompts

1. ACL 2021 [WARP: Word-level Adversarial ReProgramming](https://zhuanlan.zhihu.com/p/407144573)(Continuous Prompt开篇之作)
2. Arxiv 2021 **Stanford**[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://zhuanlan.zhihu.com/p/407144573)(Continuous Prompt用于NLG的各种任务)(将prompt用于NLG任务上)
3. Arxiv 2021 **Google**[The Power of Scale for Parameter-Efficient Prompt Tuning](https://zhuanlan.zhihu.com/p/407144573)(目前最简单的preifx training：只对input添加prefix)
4. Arxiv 2021 **DeepMind**[Multimodal Few-Shot Learning with Frozen Language Models](https://zhuanlan.zhihu.com/p/407144573)(利用图像编码器把图像作为一种动态的prefix，与文本一起送入LM中)

## Large-Pretrain-Model

1. ICML 2023 Oral [Scaling Vision Transformers to 22 Billion Parameters](https://zhuanlan.zhihu.com/p/639174050)(将ViT扩展到220亿的参数量)
2. ICML 2023 Oral [Specializing Smaller Language Models towards Multi-Step Reasoning](https://zhuanlan.zhihu.com/p/639174050)(通过蒸馏降低大模型的全面性，提升小模型的专业性)
3. ICML 2023 Oral [Pretraining Language Models with Human Preferences](https://zhuanlan.zhihu.com/p/639174050)(练一开始就纳入人类偏好而不是finetune时)
4. ICML 2023 Oral [Whose Opinions Do Language Models Reflect?](https://zhuanlan.zhihu.com/p/639174050)(LMs反映的观点与美国人口群体存在相当大的不一致)
5. ICML 2023 Oral [Mimetic Initialization of Self-Attention Layers](https://zhuanlan.zhihu.com/p/639174050)(mimetic initialization让attention在小数据集上也能训练的很好)
6. ICML 2023 Oral [Cross-Modal Fine-Tuning: Align then Refine](https://zhuanlan.zhihu.com/p/639174050)(跨模态微调框架，将单个大规模预训练模型的适用性扩展到多样的模态)
7. ICML 2023 Oral [Evaluating Self-Supervised Learning via Risk Decomposition](https://zhuanlan.zhihu.com/p/639174050)(综合指标评估自监督学习（SSL）的性能)
8. ICML 2023 [XTab: Cross-table Pretraining for Tabular Transformers](https://zhuanlan.zhihu.com/p/614276528)(在各个领域的不同数据表上进行跨表预训练)
9. ICML 2023 [Multi-Environment Pretraining Enables Transfer to Action Limited Datasets](https://zhuanlan.zhihu.com/p/614276528)(解决大规模训练模型所需的序列决策数据往往缺乏标注的动作信息)
10. ICML 2023 [Why do Nearest Neighbor Language Models Work?](https://zhuanlan.zhihu.com/p/614276528)(分析kNN-LM与传统LM的不同维度来回答kNN-LM为什么好)
11. ICML 2023 [PaLM-E: An Embodied Multimodal Language Model](https://zhuanlan.zhihu.com/p/614276528)(将真实世界中的连续传感器模态直接融入语言模型中，从而建立单词和感知之间的联系)
12. ICML 2023 [Compositional Exemplars for In-context Learning](https://zhuanlan.zhihu.com/p/614276528)(如何改进现有的上下文示例选择方法)
13. ICML 2023 [Synthetic Prompting: Generating Chain-of-Thought Demonstrations for Large Language Models](https://zhuanlan.zhihu.com/p/614276528)(利用人工创建的一些示例来引导大型语言模型自动生成更多示例，并选择有效的示例以促进更好的推理能力。)

# Optimization/GNN/Energy/Generative/Causality/Others

## Optimization
1. ICML 2021 [An End-to-End Framework for Molecular Conformation Generation via Bilevel Programming](https://zhuanlan.zhihu.com/p/390808626)
2. NeurIPS 2021 _Deep Structural Causal Models for Tractable Counterfactual Inference_
1. ICML 2018 _Bilevel Programming for Hyperparameter Optimization and Meta-Learning_(用bi-level programming建模超参数搜索与meta-learning)
2. NeurIPS 2021 [Energy-based Out-of-distribution Detection](https://zhuanlan.zhihu.com/p/343678039)

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

## Implicit Neural Representation (INR)
1. NeurIPS 2020 (Oral)： [Implicit Neural Representations with Periodic Activation Functions](https://zhuanlan.zhihu.com/p/472942119)
2. SIGGRAPH Asia 2020： [X-Fields: Implicit Neural View-, Light- and Time-Image Interpolation](https://zhuanlan.zhihu.com/p/472942119)
3. CVPR 2021 (Oral)：[Learning Continuous Image Representation with Local Implicit Image Function](https://zhuanlan.zhihu.com/p/472942119)
4. CVPR 2021 [Adversarial Generation of Continuous Images](https://arxiv.org/abs/2011.12026)
5. NeurIPS 2021 [Learning Signal-Agnostic Manifolds of Neural Fields](https://arxiv.org/abs/2111.06387)
6. Arxiv 2021 [Generative Models as Distributions of Functions](https://arxiv.org/abs/2102.04776)


## Survey
1. [综述：基于能量的模型](https://zhuanlan.zhihu.com/p/343529491)
