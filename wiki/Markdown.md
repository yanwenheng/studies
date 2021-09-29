Markdown 备忘
===

- [插件](#插件)
    - [自动更新目录插件（VSCode）](#自动更新目录插件vscode)
- [常用操作](#常用操作)
    - [换行](#换行)
- [居中插入图片](#居中插入图片)
    - [隐藏块](#隐藏块)
    - [HTML 表格](#html-表格)
    - [Latex](#latex)


## 插件
### 自动更新目录插件（VSCode）
- 搜索插件 `Markdown All in One`
- 插入目录 `Shift+Command+P` -> `Create Table of Contents`


## 常用操作
### 换行
```markdown
<br/>
```

## 居中插入图片

<style> 
.test{width:300px; align:"center"; overflow:hidden} 
.test img{max-width:300px;_width:expression(this.width > 300 ? "300px" : this.width);} 
</style> 

- 不带链接
    ```
    <div align="center"><img src="./_assets/xxx.png" height="300" /></div>
    ```
- 带链接
    ```
    <div align="center"><a href=""><img src="./_assets/xxx.png" height="300" /></a></div>
    ```
- `height`用于控制图片的大小，一般不使用，图片会等比例缩放；


### 隐藏块
```
<details><summary><b>示例：动态序列（点击展开）</b></summary> 

// 代码块，注意上下都要保留空行

</details>
<br/> <!-- 如果间隔太小，可以加一个空行 -->
```


### HTML 表格
```
<table style="width:80%; table-layout:fixed;">
    <tr>
        <th align="center">普通卷积</td>
        <th align="center">空洞卷积</td>
    </tr>
    <tr>
        <td><img width="250px" src="./res/no_padding_no_strides.gif"></td>
        <td><img width="250px" src="./res/dilation.gif"></td>
    </tr>
</table>
```

### Latex
> markdown 专用
- 在 markdown 内使用：行内使用 `$` 包围，独立行使用 `$$` 包围

**小技巧：参考和引用**

$[1]$ [xxx](xxx) <br/>

引用$^{[1]}$