My Code Lab
---

<font color="LightGrey"><i> `This README is Auto-generated` </i></font>

<details><summary><b> Image Utils <a href="#image-utils">¶</a></b></summary>

- [`ImageCheck`: 图片完整性检查](#imagecheck-图片完整性检查)
- [`get_real_ext`: 获取图像文件的真实后缀](#get_real_ext-获取图像文件的真实后缀)

</details>

<details><summary><b> NLP Utils <a href="#nlp-utils">¶</a></b></summary>

- [`BertTokenizer`: Bert 分词器](#berttokenizer-bert-分词器)
- [`split`: 将数据按比例切分](#split-将数据按比例切分)
- [`ner_result_parse`: NER 结果解析（基于 BIO 格式）](#ner_result_parse-ner-结果解析基于-bio-格式)

</details>

<details><summary><b> Python Utils <a href="#python-utils">¶</a></b></summary>

- [`simple_argparse`: 一个简化版 argparse](#simple_argparse-一个简化版-argparse)

</details>

<details><summary><b> Python 自定义数据结构 <a href="#python-自定义数据结构">¶</a></b></summary>

- [`ArrayDict`: 数组字典，支持 slice](#arraydict-数组字典支持-slice)
- [`ValueArrayDict`: 数组字典，支持 slice，且操作 values](#valuearraydict-数组字典支持-slice且操作-values)
- [`BunchDict`: 基于 dict 实现 Bunch 模式](#bunchdict-基于-dict-实现-bunch-模式)
- [`ConfigDict`: 配置字典（基于 BunchDict）](#configdict-配置字典基于-bunchdict)

</details>

<details><summary><b> Pytorch Loss <a href="#pytorch-loss">¶</a></b></summary>

- [`ContrastiveLoss`: 对比损失（默认距离函数为欧几里得距离）](#contrastiveloss-对比损失默认距离函数为欧几里得距离)
- [`CrossEntropyLoss`: 交叉熵](#crossentropyloss-交叉熵)
- [`TripletLoss`: Triplet 损失，常用于无监督学习、few-shot 学习](#tripletloss-triplet-损失常用于无监督学习few-shot-学习)

</details>

<details><summary><b> Pytorch Models <a href="#pytorch-models">¶</a></b></summary>

- [`DualNet`: 双塔结构](#dualnet-双塔结构)
- [`SiameseNet`: 孪生网络，基于双塔结构](#siamesenet-孪生网络基于双塔结构)
- [`SimCSE`: SimCSE](#simcse-simcse)
- [`Bert`: Bert by Pytorch](#bert-bert-by-pytorch)

</details>

<details><summary><b> Pytorch Utils <a href="#pytorch-utils">¶</a></b></summary>

- [`ToyDataLoader`: 一个简单的 DataLoader](#toydataloader-一个简单的-dataloader)
- [`DictTensorDataset`: 字典形式的 Dataset](#dicttensordataset-字典形式的-dataset)
- [`Trainer`: 一个简单的 Pytorch Trainer](#trainer-一个简单的-pytorch-trainer)

</details>

---

## Image Utils

### `ImageCheck`: 图片完整性检查
> [source](my/vision/image_check.py#L21)

<details><summary><b> Intro & Example </b></summary>

```python
图片完整性检查

Examples:
    >>> img = r'./_test_data/pok.jpg'
    >>> ImageCheck.is_complete(img)

```

</details>


### `get_real_ext`: 获取图像文件的真实后缀
> [source](my/vision/image_utils.py#L21)

<details><summary><b> Intro & Example </b></summary>

```python
获取图像文件的真实后缀
如果不是图片，返回后缀为 None
该方法不能判断图片是否完整

Args:
    image_path:
    return_is_same: 是否返回 `is_same`

Returns:
    ext_real, is_same
    真实后缀，真实后缀与当前后缀是否相同
    如果当前文件不是图片，则 ext_real 为 None
```

</details>


## NLP Utils

### `BertTokenizer`: Bert 分词器
> [source](my/nlp/bert_tokenizer.py#L233)

<details><summary><b> Intro & Example </b></summary>

```python
Bert 分词器

Examples:
    >>> text = '我爱python，我爱编程；I love python, I like programming. Some unkword'

    # WordPiece 切分
    >>> tokens = tokenizer.tokenize(text)
    >>> assert [tokens[2], tokens[-2], tokens[-7]] == ['python', '##nk', 'program']

    # 模型输入
    >>> token_ids, segment_ids, masks = tokenizer.encode(text)
    >>> assert token_ids[:6] == [101, 2769, 4263, 9030, 8024, 2769]
    >>> assert segment_ids == [0] * len(token_ids)

    # 句对模式
    >>> txt1 = '我爱python'
    >>> txt2 = '我爱编程'
    >>> token_ids, segment_ids, masks = tokenizer.encode(txt1, txt2)
    >>> assert token_ids == [101, 2769, 4263, 9030, 102, 2769, 4263, 5356, 4923, 102]
    >>> assert segment_ids == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

```

</details>


### `split`: 将数据按比例切分
> [source](my/nlp/data_utils.py#L61)

<details><summary><b> Intro & Example </b></summary>

```python
将数据按比例切分

Args:
    *arrays:
    split_size: 切分比例，采用向上取整：ceil(6*0.3) = 2
    random_seed: 随机数种子
    shuffle: 是否打乱

Examples:
    >>> data = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
    >>> xt, xv = split(*data, split_size=0.3, shuffle=False)
    >>> xt
    [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
    >>> xv
    [[5, 6, 7], [5, 6, 7], [5, 6, 7]]
    
Returns:
    x_train, x_val =  split(x)
    (a_train, b_train, c_train), (a_val, b_train, c_train) = split(a, b, c)
```

</details>


### `ner_result_parse`: NER 结果解析（基于 BIO 格式）
> [source](my/nlp/ner_utils.py#L22)

<details><summary><b> Intro & Example </b></summary>

```python
NER 结果解析（基于 BIO 格式）

Examples:
    >>> _label_id2name = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-LOC', 4: 'I-LOC'}
    >>> _tokens = list('你知道小明生活在北京吗？')
    >>> _labels = list(map(int, '000120003400'))
    >>> ner_result_parse(_tokens, _labels, _label_id2name)
    [['PER', '小明', (3, 4)], ['LOC', '北京', (8, 9)]]

    >>> _tokens = list('小明生活在北京')  # 测试头尾是否正常
    >>> _labels = list(map(int, '1200034'))
    >>> ner_result_parse(_tokens, _labels, label_id2name=_label_id2name)
    [['PER', '小明', (0, 1)], ['LOC', '北京', (5, 6)]]

    >>> _tokens = list('明生活在北京')  # 明: I-PER
    >>> _labels = list(map(int, '200034'))
    >>> ner_result_parse(_tokens, _labels, label_id2name=_label_id2name)
    [['LOC', '北京', (4, 5)]]

    >>> _tokens = list('小明生活在北')
    >>> _labels = list(map(int, '120003'))  # 北: B-LOC
    >>> ner_result_parse(_tokens, _labels, label_id2name=_label_id2name)
    [['PER', '小明', (0, 1)], ['LOC', '北', (5, 5)]]

Args:
    tokens:
    labels:
    token_id2name:
    label_id2name:

Returns:
    example: [['小明', 'PER', (3, 4)], ['北京', 'LOC', (8, 9)]]
```

</details>


## Python Utils

### `simple_argparse`: 一个简化版 argparse
> [source](my/python/custom/simple_argparse.py#L25)

<details><summary><b> Intro & Example </b></summary>

```python
一个简化版 argparse

不需要预先设置字段，严格按照 `--a A` 一组的方式自动提取，
    其中 A 部分会调用 eval()，某种程度上比自带的 argparse 更强大

Examples:
    >>> from my.python.custom import ConfigDict, simple_argparse
    >>> sys.argv = ['xxx.py', '--a', 'A', '--b', '1', '--c', '3.14', '--d', '[1,2]', '--e', '"[1,2]"']
    >>> simple_argparse()
    {'a': 'A', 'b': 1, 'c': 3.14, 'd': [1, 2], 'e': '[1,2]'}
    >>> _args = ConfigDict(x=1, b=20)
    >>> simple_argparse(_args)
    {'x': 1, 'b': 1, 'a': 'A', 'c': 3.14, 'd': [1, 2], 'e': '[1,2]'}
    >>> sys.argv = ['xxx.py']
    >>> simple_argparse(_args)
    {'x': 1, 'b': 1, 'a': 'A', 'c': 3.14, 'd': [1, 2], 'e': '[1,2]'}
    >>> sys.argv = ['xxx.py', '-a', 'A']
    >>> simple_argparse()
    Traceback (most recent call last):
        ...
    AssertionError: `-a` should starts with "--"

```

</details>


## Python 自定义数据结构

### `ArrayDict`: 数组字典，支持 slice
> [source](my/python/custom/special_dict.py#L39)

<details><summary><b> Intro & Example </b></summary>

```python
数组字典，支持 slice

Examples:
    >>> d = ArrayDict(a=1, b=2)
    >>> d
    ArrayDict([('a', 1), ('b', 2)])
    >>> d['a']
    1
    >>> d[1]
    ArrayDict([('b', 2)])
    >>> d['c'] = 3
    >>> d[0] = 100
    Traceback (most recent call last):
        ...
    TypeError: ArrayDict cannot use `int` as key.
    >>> d[1: 3]
    ArrayDict([('b', 2), ('c', 3)])
    >>> print(*d)
    a b c
    >>> d.setdefault('d', 4)
    4
    >>> print(d)
    ArrayDict([('a', 1), ('b', 2), ('c', 3), ('d', 4)])
    >>> d.pop('a')
    1
    >>> d.update({'b': 20, 'c': 30})
    >>> def f(**d): print(d)
    >>> f(**d)
    {'b': 20, 'c': 30, 'd': 4}

```

</details>


### `ValueArrayDict`: 数组字典，支持 slice，且操作 values
> [source](my/python/custom/special_dict.py#L100)

<details><summary><b> Intro & Example </b></summary>

```python
数组字典，支持 slice，且操作 values

Examples:
    >>> d = ValueArrayDict(a=1, b=2)
    >>> d
    ValueArrayDict([('a', 1), ('b', 2)])
    >>> assert d[1] == 2
    >>> d['c'] = 3
    >>> assert d[2] == 3
    >>> d[1:]
    (2, 3)
    >>> print(*d)  # 注意打印的是 values
    1 2 3
    >>> del d['a']
    >>> d.update({'a':10, 'b': 20})
    >>> d
    ValueArrayDict([('b', 20), ('c', 3), ('a', 10)])

```

</details>


### `BunchDict`: 基于 dict 实现 Bunch 模式
> [source](my/python/custom/special_dict.py#L166)

<details><summary><b> Intro & Example </b></summary>

```python
基于 dict 实现 Bunch 模式

行为上类似于 argparse.Namespace，但可以使用 dict 的方法，更通用

Examples:
    >>> c = BunchDict(a=1, b=2)
    >>> c
    {'a': 1, 'b': 2}
    >>> c.c = 3
    >>> c
    {'a': 1, 'b': 2, 'c': 3}
    >>> dir(c)
    ['a', 'b', 'c']
    >>> assert 'a' in c
    >>> del c.a
    >>> assert 'a' not in c

    >>> x = BunchDict(d=4, e=c)
    >>> x
    {'d': 4, 'e': {'b': 2, 'c': 3}}
    >>> z = {'d': 4, 'e': {'a': 1, 'b': 2, 'c': 3}}
    >>> y = BunchDict.from_dict(z)
    >>> y
    {'d': 4, 'e': {'a': 1, 'b': 2, 'c': 3}}

References:
    - bunch（pip install bunch）
```

</details>


### `ConfigDict`: 配置字典（基于 BunchDict）
> [source](my/python/custom/special_dict.py#L248)

<details><summary><b> Intro & Example </b></summary>

```python
配置字典（基于 BunchDict）

在 BunchDict 基础上添加了 save/load 等操作。

Examples:
    # _TestConfig 继承自 BaseConfig，并对配置项设置默认值
    >>> class _TestConfig(ConfigDict):
    ...     def __init__(self, **config_items):
    ...         from datetime import datetime
    ...         self.a = 1
    ...         self.b = datetime(2012, 1, 1)  # 注意是一个特殊对象，默认 json 是不支持的
    ...         super(_TestConfig, self).__init__(**config_items)

    >>> args = _TestConfig()
    >>> assert args.a == 1  # 默认值
    >>> args.a = 10  # 修改值
    >>> assert args.a == 10  # 自定义值

    >>> args = _TestConfig(a=10)  # 创建时修改
    >>> assert args.a == 10

    # 添加默认中不存的配置项
    >>> args.c = 3  # 默认中没有的配置项（不推荐，建议都定义在继承类中，并设置默认值）
    >>> assert args.c == 3
    >>> print(args)  # 注意 'b' 保存成了特殊形式
    _TestConfig: {
        "a": 10,
        "b": "datetime.datetime(2012, 1, 1, 0, 0)__@AnyEncoder@__gASVKgAAAAAAAACMCGRhdGV0aW1llIwIZGF0ZXRpbWWUk5RDCgfcAQEAAAAAAACUhZRSlC4=",
        "c": 3
    }

    # 保存配置到文件
    >>> fp = r'./-test/test_save_config.json'
    >>> os.makedirs(os.path.dirname(fp), exist_ok=True)
    >>> args.save(fp)  # 保存
    >>> x = _TestConfig.load(fp)  # 重新加载
    >>> assert x.dict == args.dict
    >>> _ = os.system('rm -rf ./-test')

```

</details>


## Pytorch Loss

### `ContrastiveLoss`: 对比损失（默认距离函数为欧几里得距离）
> [source](my/pytorch/loss/contrastive.py#L49)

<details><summary><b> Intro & Example </b></summary>

```python
对比损失（默认距离函数为欧几里得距离）
```

</details>


### `CrossEntropyLoss`: 交叉熵
> [source](my/pytorch/loss/cross_entropy.py#L214)

<details><summary><b> Intro & Example </b></summary>

```python
交叉熵

TODO: 实现 weighted、smooth

Examples:
    >>> logits = torch.rand(5, 5)
    >>> labels = torch.arange(5)
    >>> probs = torch.softmax(logits, dim=-1)
    >>> onehot_labels = F.one_hot(labels)
    >>> my_ce = CrossEntropyLoss(reduction='none', onehot_label=True)
    >>> ce = nn.CrossEntropyLoss(reduction='none')
    >>> assert torch.allclose(my_ce(probs, onehot_labels), ce(logits, labels), atol=1e-5)

```

</details>


### `TripletLoss`: Triplet 损失，常用于无监督学习、few-shot 学习
> [source](my/pytorch/loss/triplet.py#L77)

<details><summary><b> Intro & Example </b></summary>

```python
Triplet 损失，常用于无监督学习、few-shot 学习

Examples:
    >>> anchor = torch.randn(100, 128)
    >>> positive = torch.randn(100, 128)
    >>> negative = torch.randn(100, 128)

    # my_tl 默认 euclidean_distance_nosqrt
    >>> tl = TripletLoss(margin=2., reduction='none')
    >>> tld = nn.TripletMarginWithDistanceLoss(distance_function=euclidean_distance_nosqrt,
    ...                                        margin=2., reduction='none')
    >>> assert torch.allclose(tl(anchor, positive, negative), tld(anchor, positive, negative), atol=1e-5)

    # 自定义距离函数
    >>> from my.pytorch.backend.distance_fn import cosine_distance
    >>> my_tl = TripletLoss(distance_fn=cosine_distance, margin=0.5, reduction='none')
    >>> tl = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=0.5, reduction='none')
    >>> assert torch.allclose(my_tl(anchor, positive, negative), tl(anchor, positive, negative), atol=1e-5)

```

</details>


## Pytorch Models

### `DualNet`: 双塔结构
> [source](my/studies/code/pytorch_models/modules/advance/dual.py#L25)

<details><summary><b> Intro & Example </b></summary>

```python
双塔结构
```

</details>


### `SiameseNet`: 孪生网络，基于双塔结构
> [source](my/studies/code/pytorch_models/modules/advance/siamese.py#L27)

<details><summary><b> Intro & Example </b></summary>

```python
孪生网络，基于双塔结构
```

</details>


### `SimCSE`: SimCSE
> [source](my/studies/code/pytorch_models/modules/advance/sim_cse.py#L30)

<details><summary><b> Intro & Example </b></summary>

```python
SimCSE

References: https://github.com/princeton-nlp/SimCSE
```

</details>


### `Bert`: Bert by Pytorch
> [source](my/studies/code/pytorch_models/modules/transformer/bert.py#L136)

<details><summary><b> Intro & Example </b></summary>

```python
Bert by Pytorch

Examples:
    >>> # My bert 1 (default)
    >>> bert = get_bert_pretrained(return_tokenizer=False)

    # 输出测试
    >>> from my.nlp.bert_tokenizer import tokenizer
    >>> s = '我爱机器学习'
    >>> tokens_ids, segments_ids, masks = tokenizer.batch_encode([s], max_len=10, convert_fn=torch.as_tensor)

    # transformers Bert
    >>> from transformers import BertModel
    >>> model = BertModel.from_pretrained('bert-base-chinese')
    >>> model.config.output_hidden_states = True
    >>> o_pt = model(tokens_ids, masks, segments_ids)

    >>> o_my = bert(tokens_ids, segments_ids)
    >>> # cls embedding
    >>> assert torch.allclose(o_pt.pooler_output, o_my[0], atol=1e-5)
    >>> # last_hidden_state
    >>> assert torch.allclose(o_pt.last_hidden_state, o_my[1], atol=1e-5)
    >>> # all_hidden_state
    >>> assert torch.allclose(torch.cat(o_pt.hidden_states), torch.cat(o_my[-1]), atol=1e-5)
```

</details>


## Pytorch Utils

### `ToyDataLoader`: 一个简单的 DataLoader
> [source](my/pytorch/train/data_utils.py#L38)

<details><summary><b> Intro & Example </b></summary>

```python
一个简单的 DataLoader

简化中间创建 Dataset 的过程，直接从数据（tensor/list/ndarray）创建 DataLoader

Examples:
    >>> x = y = torch.as_tensor([1,2,3,4,5])
    >>> # 返回 tuple
    >>> dl = ToyDataLoader([x, y], batch_size=3, shuffle=False)
    >>> for batch in dl:
    ...     print(batch)
    [tensor([1, 2, 3]), tensor([1, 2, 3])]
    [tensor([4, 5]), tensor([4, 5])]
    >>> # 返回 dict
    >>> dl = ToyDataLoader({'x': x, 'y': y}, batch_size=3, shuffle=False)
    >>> for batch in dl:
    ...     print(batch)
    {'x': tensor([1, 2, 3]), 'y': tensor([1, 2, 3])}
    {'x': tensor([4, 5]), 'y': tensor([4, 5])}
```

</details>


### `DictTensorDataset`: 字典形式的 Dataset
> [source](my/pytorch/train/data_utils.py#L77)

<details><summary><b> Intro & Example </b></summary>

```python
字典形式的 Dataset

使用本类生成 DataLoader 时，可以返回 dict 类型的 batch

Examples:
    >>> x = y = torch.as_tensor([1,2,3,4,5])
    >>> ds = DictTensorDataset(x=x, y=y)
    >>> len(ds)
    5
    >>> dl = DataLoader(ds, batch_size=3)
    >>> for batch in dl: print(batch)
    {'x': tensor([1, 2, 3]), 'y': tensor([1, 2, 3])}
    {'x': tensor([4, 5]), 'y': tensor([4, 5])}
    >>> len(dl)
    2

References:
    - torch.utils.data.TensorDataset
    - huggingface/datasets.arrow_dataset.Dataset
```

</details>


### `Trainer`: 一个简单的 Pytorch Trainer
> [source](my/pytorch/train/trainer.py#L47)

<details><summary><b> Intro & Example </b></summary>

```python
一个简单的 Pytorch Trainer

Examples:
    >>> # 参考 code/examples/pytorch/train_ner_bert_crf.py
```

</details>
