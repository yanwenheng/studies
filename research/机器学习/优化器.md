优化器（Optimizer）
===

Index
---
<!-- TOC -->

- [AdaFactor](#adafactor)

<!-- /TOC -->


## AdaFactor
> 【解读】[AdaFactor优化器浅析（附开源实现） - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/7302)

**小结**
- AdaFactor 具有自适应学习率的特性，但比 RMSProp 更省显存，并针对性地解决了 Adam 的一些缺陷。
- 主要用于大模型预训练，可以使用更大的 batch，下游任务微调效果可能不如直接使用 Adam。