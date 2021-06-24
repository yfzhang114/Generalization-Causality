domain generalization， OOD问题的阅读清单，链接为笔记，没有链接就是还没看，欢迎大家commit

# 2021
1. Arxiv: [Towards a Theoretical Framework of Out-of-Distribution Generalization](https://zhuanlan.zhihu.com/p/382608823) （新理论）
2. Arxiv(**Yoshua Bengio**) Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization
3. Arxiv Delving Deep into the Generalization of Vision Transformers under Distribution Shifts (视觉transformer的泛化性讨论)
4. ICML Oral： [Can Subnetwork Structure be the Key to Out-of-Distribution Generalization?](https://zhuanlan.zhihu.com/p/382608823) （彩票模型寻找模型中泛化能力更强的小模型）
5. ICML Oral：[Domain Generalization using Causal Matching](https://zhuanlan.zhihu.com/p/382608823) (contrastive loss特征对齐+特征不变性约束)
6. ICML Spotlight: [Environment Inference for Invariant Learning](https://zhuanlan.zhihu.com/p/382608823)
7. ICLR Poster: [Understanding the failure modes of out-of-distribution generalization](https://zhuanlan.zhihu.com/p/382608823) （OOD失败的两种原因）
8. ICLR Poster In Search of Lost Domain Generalization (没有model selection的方法不是好方法，如何根据验证集选择模型？)
9. CVPR Oral: [Reducing Domain Gap by Reducing Style Bias](https://zhuanlan.zhihu.com/p/382608823) (channel-wise 均值作为图像风格，减少CNN对风格的依赖)
10. AISTATS Linear Regression Games: Convergence Guarantees to Approximate Out-of-Distribution Solutions

# 2020
1. Arxiv [I-SPEC: An End-to-End Framework for Learning Transportable, Shift-Stable Models](https://zhuanlan.zhihu.com/p/288980706)(将Domain Adaptation看作是因果图推理问题)
2. NeurIPS: [Energy-based Out-of-distribution Detection](https://zhuanlan.zhihu.com/p/343678039)(使用能量模型检测OOD样本)
3. Arxiv [Invariant Risk Minimization](https://zhuanlan.zhihu.com/p/273209891) (奠基之作，跳出经验风险最小化--不变风险最小化)
4. ICLR Poster [The Risks of Invariant Risk Minimization](https://zhuanlan.zhihu.com/p/273209891) (不变风险最小化的缺陷:域数目过少IRM即失败)

# OLD but Important
1. ICCV 2017 [CCSA--Unified Deep Supervised Domain Adaptation and Generalization](https://blog.csdn.net/Adupanfei/article/details/85165667) (对比损失对齐源域目标域样本空间)

# 综述
1. [Domain Adaptation基础概念与相关文章解读](https://zhuanlan.zhihu.com/p/272508224)
