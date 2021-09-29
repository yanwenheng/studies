电脑操作相关
===

<!-- TOC -->

- [Mac](#mac)
    - [常用软件安装](#常用软件安装)
        - [brew（软件安装）](#brew软件安装)
        - [expect（远程登陆）](#expect远程登陆)
    - [常用操作](#常用操作)
        - [查看本机地址](#查看本机地址)
        - [关闭首字母大写](#关闭首字母大写)
- [Windows](#windows)

<!-- /TOC -->


## Mac

### 常用软件安装

#### brew（软件安装）
> [The Missing Package Manager for macOS (or Linux) — Homebrew](https://brew.sh/)

#### expect（远程登陆）
> `brew install expect`

- 远程登陆
    ```bash
    #!/usr/bin/expect
    set timeout 10

    set username "xxx"
    set password "xxx"

    spawn ssh ${username}@jumper.sankuai.com
    expect "*password*"
    send "${password}\r"

    interact
    ```
- 注意：运行时直接`jumper.sh`，而不要 `sh jumper.sh`，否则会报错。


### 常用操作

#### 查看本机地址
```
# 查看位置：系统偏好设置 -> 共享 -> 远程登录
# 形如 `user@ip`

# 远程复制（服务器端操作）
scp [文件位置] user@ip:[本地位置]
```

#### 关闭首字母大写
> 系统偏好设置 -> 键盘 -> 文本


## Windows