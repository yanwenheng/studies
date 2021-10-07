studies
===

### 其他仓库
- [imhuay/bert_by_keras](https://github.com/imhuay/bert_by_keras)

---

<font color="LightGrey"><i> `The following is Auto-generated` </i></font>

---

Repo Index
---

- [Algorithm Studies](#algorithm-studies)
- [My Code Lab](#my-code-lab)

---

Algorithm Studies
---

<details><summary><b> 专题-二叉树(树) <a href="algorithm/专题-二叉树(树).md">¶</a></b></summary>

- [路径总和 III](algorithm/专题-二叉树(树).md#路径总和-iii)

</details>

<details><summary><b> 专题-前缀和 <a href="algorithm/专题-前缀和.md">¶</a></b></summary>

- [路径总和 III](algorithm/专题-前缀和.md#路径总和-iii)

</details>

<details><summary><b> 专题-双指针(滑动窗口) <a href="algorithm/专题-双指针(滑动窗口).md">¶</a></b></summary>

- [三数之和（3Sum）](algorithm/专题-双指针(滑动窗口).md#三数之和3sum)
- [最接近的三数之和（3Sum Closest）](algorithm/专题-双指针(滑动窗口).md#最接近的三数之和3sum-closest)

</details>

<details><summary><b> 专题-哈希表 <a href="algorithm/专题-哈希表.md">¶</a></b></summary>

- [两数之和](algorithm/专题-哈希表.md#两数之和)

</details>

<details><summary><b> 专题-深度优先搜索(递归) <a href="algorithm/专题-深度优先搜索(递归).md">¶</a></b></summary>

- [路径总和 III](algorithm/专题-深度优先搜索(递归).md#路径总和-iii)

</details>

<details><summary><b> 专题-递归(迭代) <a href="algorithm/专题-递归(迭代).md">¶</a></b></summary>

- [合并两个有序链表](algorithm/专题-递归(迭代).md#合并两个有序链表)

</details>

<details><summary><b> 专题-链表 <a href="algorithm/专题-链表.md">¶</a></b></summary>

- [两数相加](algorithm/专题-链表.md#两数相加)

</details>

---

My Code Lab
---

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

<details><summary><b> Pytorch Utils <a href="#pytorch-utils">¶</a></b></summary>

- [`DictTensorDataset`: 字典形式的 Dataset](#dicttensordataset-字典形式的-dataset)
- [`ToyDataLoader`: 一个简单的 DataLoader](#toydataloader-一个简单的-dataloader)

</details>

---

## NLP Utils

### `BertTokenizer`: Bert 分词器
> [source](code/my/nlp/bert_tokenizer.py#L233)

**Examples:**
```python
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

### `split`: 将数据按比例切分
> [source](code/my/nlp/data_utils.py#L54)

**Examples:**
```python
>>> data = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
>>> xt, xv = split(*data, split_size=0.3, shuffle=False)
>>> xt
[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
>>> xv
[[5, 6, 7], [5, 6, 7], [5, 6, 7]]
```

### `ner_result_parse`: NER 结果解析（基于 BIO 格式）
> [source](code/my/nlp/ner_utils.py#L22)

**Examples:**
```python
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
```

## Python Utils

### `simple_argparse`: 一个简化版 argparse
> [source](code/my/python/custom/simple_argparse.py#L25)

不需要预先设置字段，严格按照 `--a A` 一组的方式自动提取，<br>
    其中 A 部分会调用 eval()，某种程度上比自带的 argparse 更强大

**Examples:**
```python
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

## Python 自定义数据结构

### `ArrayDict`: 数组字典，支持 slice
> [source](code/my/python/custom/special_dict.py#L39)

**Examples:**
```python
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

### `ValueArrayDict`: 数组字典，支持 slice，且操作 values
> [source](code/my/python/custom/special_dict.py#L100)

**Examples:**
```python
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

### `BunchDict`: 基于 dict 实现 Bunch 模式
> [source](code/my/python/custom/special_dict.py#L166)

行为上类似于 argparse.Namespace，但可以使用 dict 的方法，更通用

**Examples:**
```python
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
```

### `ConfigDict`: 配置字典（基于 BunchDict）
> [source](code/my/python/custom/special_dict.py#L248)

在 BunchDict 基础上添加了 save/load 等操作。

**Examples:**
```python
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

## Pytorch Utils

### `DictTensorDataset`: 字典形式的 Dataset
> [source](code/my/pytorch/pipeline/dataset.py#L42)

使用本类生成 DataLoader 时，可以返回 dict 类型的 batch

**Examples:**
```python
>>> x = y = torch.as_tensor([1,2,3,4,5])
>>> ds = DictTensorDataset(x=x, y=y)
>>> len(ds)
5
>>> dl = DataLoader(ds, batch_size=3)
>>> for batch in dl: print(batch)
{'x': tensor([1, 2, 3]), 'y': tensor([1, 2, 3])}
{'x': tensor([4, 5]), 'y': tensor([4, 5])}
```

### `ToyDataLoader`: 一个简单的 DataLoader
> [source](code/my/pytorch/pipeline/dataset.py#L82)

简化中间创建 Dataset 的过程，直接从数据（tensor/list/ndarray）创建 DataLoader

**Examples:**
```python
>>> x = y = torch.as_tensor([1,2,3,4,5])

# 返回 tuple
>>> dl = ToyDataLoader([x, y], batch_size=3, shuffle=False)
>>> for batch in dl: print(batch)
[tensor([1, 2, 3]), tensor([1, 2, 3])]
[tensor([4, 5]), tensor([4, 5])]

# 返回 dict
>>> dl = ToyDataLoader({'x': x, 'y': y}, batch_size=3, shuffle=False)
>>> for batch in dl: print(batch)
{'x': tensor([1, 2, 3]), 'y': tensor([1, 2, 3])}
{'x': tensor([4, 5]), 'y': tensor([4, 5])}
```
