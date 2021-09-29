yaml 相关
===

Index
---
<!-- TOC -->

- [基本语法](#基本语法)
- [常用数据类型](#常用数据类型)
    - [字典（键值对）](#字典键值对)
    - [列表（数组）](#列表数组)
    - [纯量](#纯量)
- [引用](#引用)

<!-- /TOC -->


## 基本语法
> [YAML 入门教程 | 菜鸟教程](https://www.runoob.com/w3cnote/yaml-intro.html)
- 大小写敏感；
- 通过缩进表示层级关系；
    - 也可以使用 `key: {key1: value1, key2: value2, ...}`
- 缩进只使用**空格**，而不是 tab；
    - 缩进的空格数不重要，只要相同层级的元素左对齐即可；
- '#'表示注释；
- 键值对使用冒号结构表示 `key: value`，冒号后面要加**至少一个空格**；


## 常用数据类型

### 字典（键值对）
```yaml
dict1:
    a: A
    b: B
    c: C

dict2: {x: 1, y: 2, z: 3}
```

### 列表（数组）
```yaml
list1:
    - A
    - B
    - C 

list2: [1, 2, 3]

list_of_dict:
    -
        id: 1
        name: A
    -
        id: 2
        name: B
```

### 纯量
```yaml
# 布尔值
boolean: 
    - True  # true, TRUE 都可以
    - False  # false, FALSE 都可以

# 整型
int: 123

# 浮点数
float: 
    - 3.14
    - 1.0e-5  # OK，科学计数法必须带有小数点
    - 1.e-5  # OK
    - 1e-5  # ERR，这种会识别成字符串

# 字符串
string: 
    - abc
    - "Hello World"
    - Hello "Python"
    - "I'm xxx"
```


## 引用
> 不常用
```yaml
a: &a
    x: 1
    y: 2

m:
    n: 3
    <<: *a

# 等价于
a: &a
    x: 1
    y: 2

m:
    n: 3
    x: 1
    y: 2
```
