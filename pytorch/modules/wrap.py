#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Time:
    2021-07-11 11:20 上午
    
Author:
    huayang
    
Subject:
    
"""

from typing import Callable

import torch
import torch.nn as nn

from torch import Tensor


def default_encode_wrapper(encoder, inputs):
    """"""
    if isinstance(inputs, Tensor):
        return encoder(inputs)
    elif isinstance(inputs, list):
        return encoder(*inputs)
    else:
        return encoder(**inputs)


class EncoderWrapper(nn.Module):
    """ Encoder 包装模块
        用于一些网络结构，网络本身可以使用不同的 Encoder 来生成输入下游的 embedding，比如孪生网络；

    Examples:

        示例1：单独使用，相当于对给定模型的输出通过`encode_wrapper`做一次包装；
            在本例中，即对 bert 的第二个输出计算的第二维均值（句向量），输出等价于直接使用 `encode_wrapper(encoder, inputs)`；
            使用包装的好处是可以调用`nn.Module`的相关方法，比如模型保存等。
        ```python
        from my.pytorch.modules.transformer.bert import get_bert_pretrained

        bert, tokenizer = get_bert_pretrained(return_tokenizer=True)
        encode_wrapper = lambda _e, _i: _e(*_i)[1].mean(1)
        test_encoder = EncoderWrapper(bert, encode_wrapper)

        ss = ['测试1', '测试2']
        inputs = tokenizer.batch_encode(ss, max_len=10)
        o = test_encoder(inputs)
        print(o.shape)  # [2, 768]
        ```

        示例2：继承使用，常用于一些框架中，框架内的 Encoder 可以任意替换
            本例为一个常见的孪生网络结构，通过继承 `EncoderWrapper` 可以灵活替换所需的模型；
        ```python
        from my.pytorch.modules.loss import ContrastiveLoss
        from my.pytorch.backend.distance_fn import euclidean_distance
        
        class SiameseNet(EncoderWrapper):
            """"""
            def __init__(self, encoder, encoder_helper):
                """"""
                super(SiameseNet, self).__init__(encoder, encoder_helper)

                self.loss_fn = ContrastiveLoss(euclidean_distance)  # 基于欧几里得距离的对比损失
                
            def forward(self, x1, x2, labels):
                """"""
                o1 = self.encode(x1)
                o2 = self.encode(x2)
                return self.loss_fn(o1, o2, labels)
        ```
    """

    def __init__(self, encoder, encode_wrapper: Callable = None):
        """

        Args:
            encoder: 编码器
            encode_wrapper: 辅助函数接口，用于帮助调整 encoder 的输入或输出，
                比如使用 bert 作为 encoder，bert 模型的输出很多，不同任务使用的输出也不同，这是可以通过 encode_wrapper 来调整；
                函数接口如下 `def encode_wrapper(encoder, inputs)`，
                默认为 encoder 直接调用 inputs: `encode_wrapper = lambda _encoder, _inputs: _encoder(_inputs)`
        """
        super(EncoderWrapper, self).__init__()

        self.encoder = encoder
        if encode_wrapper is not None:
            self.encode_wrapper = encode_wrapper

    def forward(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def encode_wrapper(self, encoder, inputs) -> Tensor:  # noqa
        return default_encode_wrapper(encoder, inputs)

    def encode(self, inputs):
        return self.encode_wrapper(self.encoder, inputs)


class ClassificationLayer(nn.Module):
    """"""
    SINGLE_LABEL = 'single'  # 单标签
    MULTI_LABEL = 'multi'  # 多标签
    REGRESSION = 'regress'  # 回归

    def __init__(self,
                 n_classes=2,
                 hidden_size=768,
                 dropout_prob=0.1,
                 problem_type='single'):
        """

        Args:
            n_classes:
            problem_type: one of {'single', 'multi', 'regress'}
            hidden_size:
            dropout_prob:
        """
        super().__init__()

        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, n_classes)
        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()

        self.problem_type = problem_type
        if problem_type == ClassificationLayer.REGRESSION:
            # logits.shape == labels.shape
            #   num_labels > 1, shape: [B, N];
            #   num_labels = 1, shape: [B];
            self.loss_fn = nn.MSELoss()
        elif problem_type == ClassificationLayer.MULTI_LABEL:
            # logits.shape == labels.shape == [B, N];
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif problem_type == ClassificationLayer.SINGLE_LABEL:
            # logits.shape: [B, N];
            # labels: [B];
            self.loss_fn = nn.CrossEntropyLoss()  # softmax-log-NLLLoss
        else:
            raise ValueError(f'Unsupported problem_type={problem_type}')

    def forward(self, inputs, labels=None):
        """"""
        x = self.dropout(inputs)
        logits = self.dense(x)  # [B, N] <=> [batch_size, num_labels]

        if self.problem_type == ClassificationLayer.REGRESSION and self.n_classes == 1:
            logits = logits.squeeze()  # [B, 1] -> [B]

        if self.problem_type == ClassificationLayer.SINGLE_LABEL:
            probs = self.softmax(logits)
        elif self.problem_type == ClassificationLayer.MULTI_LABEL:
            probs = self.sigmoid(logits)
        else:
            probs = logits

        if labels is None:  # eval
            return probs
        else:
            loss = self.loss_fn(logits, labels)
            return probs, loss


class ClassifierWrapper(EncoderWrapper):
    """"""

    def __init__(self,
                 encoder,
                 n_classes=2,
                 hidden_size=768,
                 dropout_prob=0.1,
                 encode_wrapper=None,
                 problem_type='single'):
        """"""
        super(ClassifierWrapper, self).__init__(encoder, encode_wrapper)

        assert n_classes == 1 if problem_type == ClassificationLayer.REGRESSION else True, \
            f'n_classes must be 1 if problem_type is REGRESSION, but {n_classes}.'

        self.clf = ClassificationLayer(n_classes,
                                       hidden_size=hidden_size,
                                       dropout_prob=dropout_prob,
                                       problem_type=problem_type)

    def forward(self, inputs, labels=None):
        """"""
        x = self.encode(**inputs)
        return self.clf(x, labels=labels)


def _test():
    """"""

    def _test_EncoderWrapper():
        """"""

        class TestEncoder(EncoderWrapper):
            """"""

            def __init__(self, encoder, encoder_helper):
                """"""
                super(TestEncoder, self).__init__(encoder, encoder_helper)
                self.loss_fn = nn.CrossEntropyLoss()

            def forward(self, inputs, labels=None):
                """"""
                outputs = self.encode(inputs)

                if labels is not None:
                    loss = self.loss_fn(outputs, labels)
                    return outputs, loss

                return outputs

        from my.pytorch.modules.transformer.bert import get_bert_pretrained

        bert, tokenizer = get_bert_pretrained(return_tokenizer=True)
        encode_wrapper = lambda _e, _i: _e(*_i)[1].mean(1)  #
        test_encoder = EncoderWrapper(bert, encode_wrapper)

        ss = ['测试1', '测试2']
        inputs = tokenizer.batch_encode(ss, max_len=10, convert_fn=torch.as_tensor)
        o = test_encoder(inputs)
        assert list(o.shape) == [2, 768]

    _test_EncoderWrapper()

    def _test_TextClassification_fine_tune():
        """"""
        from my.pytorch.modules.transformer.bert import Bert
        inputs = [torch.tensor([[1, 2, 3]]), torch.tensor([[0, 0, 0]])]

        class Test(nn.Module):
            """"""

            def __init__(self, num_labels=1):
                super(Test, self).__init__()

                self.bert = Bert()
                self.clf = ClassificationLayer(num_labels)
                # self.bert.load_weights()  # 可选

            def forward(self, inputs):
                outputs = self.bert(*inputs)
                cls_embedding = outputs[0]
                ret = self.clf(cls_embedding)
                return ret

        clf = Test()  # bert 参与训练
        logits = clf(inputs)
        print(logits)
        print('state_dict size:', len(clf.state_dict()))

    _test_TextClassification_fine_tune()

    def _test_TextClassification():
        """"""
        from my.pytorch.modules import Bert
        test_inputs = [torch.tensor([[1, 2, 3]]), torch.tensor([[0, 0, 0]])]
        bert = Bert()
        outputs = bert(*test_inputs)
        inputs = outputs[0]  # cls_embedding

        with torch.no_grad():
            classifier = ClassificationLayer(n_classes=3)
            logits = classifier(inputs)
            print(logits)

        # bert 不参与训练
        print('state_dict size:', len(classifier.state_dict()))

    _test_TextClassification()


if __name__ == '__main__':
    """"""
    _test()
