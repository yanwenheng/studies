studies
===

### 其他仓库
- [imhuay/bert_by_keras](https://github.com/imhuay/bert_by_keras)

---

My Code Lab(Auto-Generation)
---

- [Python Utils](#python-utils)
    - [`simple_argparse`: 一个简化版 argparse](#simple_argparse-一个简化版-argparse)
- [Python 自定义数据结构](#python-自定义数据结构)
    - [`ArrayDict`: 数组字典，支持 slice](#arraydict-数组字典支持-slice)
    - [`ValueArrayDict`: 数组字典，支持 slice，且操作 values](#valuearraydict-数组字典支持-slice且操作-values)
    - [`BunchDict`: 基于 dict 实现 Bunch 模式](#bunchdict-基于-dict-实现-bunch-模式)
    - [`ConfigDict`: 配置字典（基于 BunchDict）](#configdict-配置字典基于-bunchdict)
- [Pytorch Utils](#pytorch-utils)
    - [`DictTensorDataset`: 字典形式的 Dataset](#dicttensordataset-字典形式的-dataset)

---

## Python Utils

### `simple_argparse`: 一个简化版 argparse
> [source code](code/my/python/custom/simple_argparse.py#L25)

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
> [source code](code/my/python/custom/special_dict.py#L39)

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
> [source code](code/my/python/custom/special_dict.py#L100)

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
> [source code](code/my/python/custom/special_dict.py#L166)

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
> [source code](code/my/python/custom/special_dict.py#L248)

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
> [source code](code/my/pytorch/pipeline/dataset.py#L42)

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
