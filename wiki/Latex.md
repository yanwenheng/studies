Latex 基础
===

- [Latex 公式](#latex-公式)

## Latex 公式
> [【LaTeX】LaTeX符号大全_Ljnoit-CSDN博客_latex 绝对值符号](https://blog.csdn.net/ljnoit/article/details/104264753)

> 在线 LaTeX 公式编辑器 http://www.codecogs.com/latex/eqneditor.php

**绝对值**

$$ \left | a-b \right |
$$

**函数名**
- 如果是预定义好的，直接 `\max(x)`，否则使用 `\operatorname{f}(x)`，示例：

$$ \operatorname{f}(x)
$$

```
-- 斜体加粗
\boldsymbol{x}

-- 期望
\mathbb{E}

-- 矩阵对齐
\begin{array}{ll}
 & \\
 & \\
\end{array}

-- 转置
^\mathsf{T}

-- 省略号
水平方向    \cdots
竖直方向    \vdots
对角线方向  \ddots

-- 按元素相乘
\circ
或
\odot

-- 右箭头
\rightarrow 
-- 左箭头
\leftarrow 

```