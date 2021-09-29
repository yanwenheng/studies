数据增强
===

### 数据扩充（显式）
    - EDA（Easy Data Augmentation）
        - 论文
            - Easy data augmentation techniques for boosting performance on text classification tasks
        - github
            - 【英文】https://github.com/jasonwei20/eda_nlp
            - 【中文】https://github.com/425776024/nlpcda/
            - 【中文】https://github.com/wac81/textda
    - UDA（Unsupervised Data Augmentation）

### 数据融合（隐式）
- mixup
    - 论文
        - 【NLP】Augmenting Data with Mixup for Sentence Classification: An Empirical Study
        - 【CV】mixup: Beyond empirical risk minimization
- Manifold Mixup 
    - 论文
        - 【CV】Manifold Mixup: Better Representations byInterpolating Hidden States
- SeqMix
    - 论文
        - Sequence-Level Mixed Sample Data Augmentation
- mixTemporal
- mixText
    - 论文
        - Linguistically-Informed Interpolation of Hidden Space for Semi-Supervised Text Classification
- 应用于 embedding 层的策略
    > 
    - 对抗攻击（Adversarial Attack，有监督场景）
    - 打乱词序（Token Shuffling，针对 Transformer 结构）
    - 裁剪（Cutoff）
    - Dropout
    

- 元学习
    - 孪生网络
        - 伪孪生网络
    - 原型网络
        - 半原型网络
        - 高斯原型网络
    - 关系网络
    - 匹配网络
    - 神经图灵机（NTM）
        - 记忆增强网络（MANN）
    - 模型无关元学习（MAML）
        - 对抗元学习（ADML, ADversarial Meta Learning）
        - CAML（Context Adaptation for Meta Learning, 上下文适应元学习）

- 主动学习(Activate Learning)
