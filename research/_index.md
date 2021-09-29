关键词索引
===
深度学习、机器学习、数据挖掘等

Index
---
<!-- TOC -->

- [机器学习](#机器学习)
    - [主动学习](#主动学习)
    - [度量学习](#度量学习)
    - [远程监督](#远程监督)
    - [模型蒸馏](#模型蒸馏)
- [深度学习](#深度学习)
    - [优化算法](#优化算法)
        - [Adam](#adam)
        - [AdamW](#adamw)
- [数据挖掘](#数据挖掘)
- [NLP 模型](#nlp-模型)
    - [开源库](#开源库)
    - [Self-Attention](#self-attention)
    - [Transformer 相关](#transformer-相关)
        - [UniLM](#unilm)
        - [MiniLM](#minilm)
    - [BERT 相关](#bert-相关)
        - [BERT 及其衍生](#bert-及其衍生)
            - [RoBERTa](#roberta)
            - [StructBERT](#structbert)
        - [BERT 应用](#bert-应用)
            - [BERT for 关键词抽取](#bert-for-关键词抽取)
                - [KeyBERT](#keybert)
            - [BERT for NER](#bert-for-ner)
            - [BERT for 小样本学习](#bert-for-小样本学习)
            - [BERT for 实体链接](#bert-for-实体链接)
                - [KnowBert](#knowbert)
- [NLP 研究方向](#nlp-研究方向)
    - [细粒度情感分析](#细粒度情感分析)
    - [对话](#对话)
    - [关键词抽取](#关键词抽取)
        - [综述 for 关键词抽取](#综述-for-关键词抽取)
    - [小样本学习（NLP）](#小样本学习nlp)
        - [数据增强（扩充）](#数据增强扩充)
        - [数据增强（融合）](#数据增强融合)
    - [实体链接](#实体链接)
        - [综述 for 实体链接](#综述-for-实体链接)
    - [纠错](#纠错)
- [开源库](#开源库-1)
    - [TensorFlow](#tensorflow)
        - [Keras](#keras)
    - [PyTorch](#pytorch)
    - [MatrixSlow](#matrixslow)

<!-- /TOC -->

## 机器学习

### 主动学习
> Active Learning
- [主动学习(Active Learning)_猪逻辑公园-CSDN博客](https://blog.csdn.net/qq_15111861/article/details/85264109)

### 度量学习
> Metric Learning、距离度量学习 (Distance Metric Learning，DML) 、相似度学习

- 【开源库】[pytorch for metric learning](https://github.com/KevinMusgrave/pytorch-metric-learning)


### 远程监督
**应用**
- 自动序列标注
- 意见实体抽取（Aspect Extraction）
- 命名实体识别（Named Entity Recognition，NER）
- 关系抽取（Relation Extraction）


### 模型蒸馏


## 深度学习

### 优化算法
> 优化器、optimizer

#### Adam
#### AdamW
> 权重衰减
- 【2019】[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- 【解读】[比较 Adam 和 AdamW - TFknight](https://www.cnblogs.com/tfknight/p/13425532.html)


## 数据挖掘


## NLP 模型

### 开源库
- [huggingface/transformers](https://github.com/huggingface/transformers)

### Self-Attention
- [从三大顶会论文看百变Self-Attention - 知乎](https://zhuanlan.zhihu.com/p/92335822)

### Transformer 相关
> [Transformer 专题](./Transformer/README.md)
- [BERT 及其变种](#bert-及其衍生)

#### UniLM
- 【Github】[microsoft/unilm: UniLM - Unified Language Model Pre-training / Pre-training for NLP and Beyond](https://github.com/microsoft/unilm)
- 【2020】[UniLMv2: Pseudo-Masked Language Models for Unified Language Model Pre-Training](https://arxiv.org/abs/2002.12804)
- 【2019】[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)
    - 【解读】[UniLM论文阅读笔记 - 知乎](https://zhuanlan.zhihu.com/p/113380840)
- 【中文】[开源啦！开源啦！UNILM中文模型开源啦！ - 知乎](https://zhuanlan.zhihu.com/p/163483660)
    - 【Github】[YunwenTechnology/Unilm](https://github.com/YunwenTechnology/Unilm)

#### MiniLM
- 【2020】[[2002.10957] MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/abs/2002.10957)


### BERT 相关
#### BERT 及其衍生
- BERT
    - 【官方源码、tf1】[google-research/bert](https://github.com/google-research/bert)
    - 【pytorch】 [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
    - 【pytorch】[huggingface/transformers/bert](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py)
    - 【tf2】[huggingface/transformers/bert](https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_tf_bert.py)

##### RoBERTa
- 【论文】
- 【解读】[RoBERTa 详解 - 知乎](https://zhuanlan.zhihu.com/p/103205929)

    
##### StructBERT
- 【论文】[StructBERT: Incorporating Language Structures into Pre-training for Deep Language Understanding](https://arxiv.org/abs/1908.04577)
- 【解读】[StructBERT解读_fengzhou-CSDN博客](https://blog.csdn.net/fengzhou_/article/details/107028168)


#### BERT 应用
##### BERT for 关键词抽取
> 利用 BERT 进行关键词抽取

- 【Github】[ibatra/BERT-Keyword-Extractor](https://github.com/ibatra/BERT-Keyword-Extractor)

###### KeyBERT
- 【Github】[MaartenGr/KeyBERT](https://github.com/MaartenGr/KeyBERT)


##### BERT for NER
- [基于BERT预训练的中文命名实体识别TensorFlow实现_macanv的专栏-CSDN博客_bert中文命名实体识别](https://blog.csdn.net/macanv/article/details/85684284)

##### BERT for 小样本学习
- [必须要GPT3吗？不，BERT的MLM模型也能小样本学习 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/7764)

##### BERT for 实体链接
###### KnowBert
> 利用知识库增强上下文词表示、实体消歧、实体链接；
- 【论文】[[1909.04164] Knowledge Enhanced Contextual Word Representations](https://arxiv.org/abs/1909.04164)


## NLP 研究方向

### 细粒度情感分析
> Aspect Based Sentiment Analysis, ABSA <br/>
> [ABSA 专题](./ABSA/README.md)

### 对话
> QA

**综述**
- 【2021】[Recent Advances in Deep Learning-based Dialogue Systems](https://arxiv.org/abs/2105.04387)



### 关键词抽取
> Keyword Extraction、关键词挖掘

- [BERT for 关键词抽取](#bert-for-关键词抽取)

#### 综述 for 关键词抽取
- [NLP关键词提取方法总结及实现-CSDN博客](https://blog.csdn.net/asialee_bird/article/details/96454544)
    > TfIdf、TextRank、LDA主题模型等；
- [「关键词」提取都有哪些方案？ - 知乎](https://www.zhihu.com/question/21104071)


**开源实现**
- 【Github】[LIAAD/yake](https://github.com/LIAAD/yake)
    > 单文档无监督关键词抽取，不支持中文
- 【Github】[aneesha/RAKE](https://github.com/aneesha/RAKE)


### 小样本学习（NLP）
> Few-Shot Learning、Zero-Shot Learning

#### 数据增强（扩充）

#### 数据增强（融合）


### 实体链接
- [BERT for 实体链接](#bert-for-实体链接)

#### 综述 for 实体链接
- 【博客】[实体链接（一） - Pelhans 的博客](http://pelhans.com/2019/08/16/kg_paper-note3/)
- 【2015】Entity Linking with a Knowledge Base:Issues, Techniques, and Solutions
- 【2020】Neural Entity Linking: A Survey of ModelsBased on Deep Learning


### 纠错

**相关文章**
- [Soft-Masked BERT：文本纠错与BERT的最新结合 - 知乎](https://zhuanlan.zhihu.com/p/144995580)

**开源实现**
- [shibing624/pycorrector: pycorrector is a toolkit for text error correction. 文本纠错，Kenlm，Seq2Seq_Attention，BERT，MacBERT，ELECTRA，ERNIE，Transformer等模型实现，开箱即用。](https://github.com/shibing624/pycorrector)


## 开源库

### TensorFlow
- 【官方文档】[TensorFlow Tutorials](https://www.tensorflow.org/tutorials)


#### Keras
- 【官方文档】[Keras API reference](https://keras.io/api/)


### PyTorch
- 【官方文档】[PyTorch Tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html)

- 【拓展库】[huggingface/accelerate](https://github.com/huggingface/accelerate)
    > 提供一个简单的 API，将与多GPU、TPU、FP16相关的样板代码抽离出来，保持其余代码不变。用户无须使用不便控制和调整的抽象类或编写、维护样板代码，就可以直接上手多GPU或TPU。

- 【高级API】[fastai/fastai: The fastai deep learning library](https://github.com/fastai/fastai)

- 【高级API】[fastnlp/fastNLP: fastNLP: A Modularized and Extensible NLP Framework. Currently still in incubation.](https://github.com/fastnlp/fastNLP)

### MatrixSlow
> 《用 python 实现深度学习框架》配套代码
- [zackchen/MatrixSlow: A simple deep learning framework in pure python for purpose of learning in DL](https://github.com/zackchen/MatrixSlow)