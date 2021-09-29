git 常用命令
===

<!-- TOC -->

- [参考](#参考)
- [基本命令](#基本命令)
- [删除分支](#删除分支)
- [推送分支](#推送分支)
- [撤回上次 commit](#撤回上次-commit)
- [删除已提交文件/文件夹](#删除已提交文件文件夹)
- [恢复已删除的文件](#恢复已删除的文件)
- [ssh-keygen 基本用法](#ssh-keygen-基本用法)
- [`git subtree` 基本使用](#git-subtree-基本使用)
    - [场景1：从主仓库分出子仓库](#场景1从主仓库分出子仓库)
    - [场景2：将子仓库添加到主仓库](#场景2将子仓库添加到主仓库)
    - [场景3：删除子仓库](#场景3删除子仓库)
    - [强制推送子仓库](#强制推送子仓库)
    - [注意事项：](#注意事项)
- [修改 commit 的 author 信息](#修改-commit-的-author-信息)
- [常用统计](#常用统计)
    - [统计 commit 次数](#统计-commit-次数)

<!-- /TOC -->

## 参考
> https://git-scm.com/book/zh

## 基本命令
```shell
# 初始化，新建本地仓库时使用
git init

# 暂存
git add <path>  # 暂存具体文件/文件夹
git add .   # 暂存新文件和被修改的文件，不包括删除的文件
git add -u  # --update，暂存已追踪的文件，即被修改的文件和被删除的文件
git add -A  # --all，全部暂存

# 提交
git commit -m <'提交信息'>
```

## 删除分支
```
# 删除本地分支（需 merge）
git branch -d [分支名]

# 删除本地分支（不需要 merge）
git branch -D [分支名]

# 删除远程分支
git push origin --delete [分支名]
```

## 推送分支
```
# 推送本地分支到远程分支
git push origin 本地分支名:远程分支名
```

## 撤回上次 commit
```
git reset --soft HEAD~1 
-- 撤回最近一次的commit（撤销commit，不撤销git add）

git reset --mixed HEAD~1 
-- 撤回最近一次的commit（撤销commit，撤销git add）

git reset --hard HEAD~1 
-- 撤回最近一次的commit（撤销commit，撤销git add，还原改动的代码）
```

## 删除已提交文件/文件夹
```
# 删除暂存区或分支上的文件，但是工作区还需要这个文件，后续会添加到 .gitignore
# 文件变为未跟踪的状态
git rm --cache <filepath>
git rm -r --cache <dirpath>


# 删除暂存区或分支上的文件，工作区也不需要这个文件
git rm <filepath>
git rm -r <dirpath>


# 不显示移除的文件，当文件夹中文件太多时使用
git rm -r -q --cache <dirpath>
```

## 恢复已删除的文件

**方法 1**：记得文件名
```shell
# 查看删除文件的 commit_id
git log -- [file]

# 恢复文件
git checkout commit_id [file]
```


## ssh-keygen 基本用法
- ssh key 是远程仓库识别用户身份的依据；

- 如果是通过 ssh 与远程仓库交互，第一次在本机执行 git 时需要先生成 ssh key，然后将**公钥**添加到远程仓库中；
    ```shell
    # 生成 ssh key
    ssh-keygen -t rsa
    # 或
    ssh-keygen -t rsa -C "邮箱地址"

    # 之后需要确认三次
    ## 第一次确认密钥的存储位置（默认是 ~/.ssh/id_rsa），可以 Enter 跳过
    ## 后两次确认密钥口令，默认留空，可以 Enter 跳过

    # 最后查看生成的密钥，添加到远程仓库
    cat ~/.ssh/id_rsa.pub
    ```


## `git subtree` 基本使用
> git subtree教程 - SegmentFault | https://segmentfault.com/a/1190000012002151

### 场景1：从主仓库分出子仓库

1. 关联子仓库与 git 地址（一般为空仓库）：`git remote add $name xxx.git`
2. 将子仓库提取到单独的分支：`git subtree split --prefix=$prefix --branch $name --rejoin`
3. 推送子仓库代码：`git subtree push --prefix=$prefix $name master --squash`

> 推荐在每次 push 子仓库代码时，都 `git subtree split --rejoin` 一次； <br/>
> 因为当主项目的 commit 变多后，再推送子项目到远程库的时候，subtree 每次都要遍历很多 commit；
>> 解决方法就是使用 split 打点，作为下次遍历的起点。解决这个问题后就可以彻底抛弃 `git submodule` 了；
>>> [git subtree使用体验_李棚的CSDN专栏](https://blog.csdn.net/huangxiaominglipeng/article/details/111195399)

### 场景2：将子仓库添加到主仓库

1. 关联子仓库与 git 地址：`git remote add $name xxx.git`
2. 设置子仓库路径（第一次执行时会自动拉取代码）：`git subtree add --prefix=$prefix $name master --squash`
3. 拉取子仓库代码：`git subtree pull --prefix=$prefix $name master --squash`

### 场景3：删除子仓库
1. 切断子仓库关联：`git remote remove $name`
2. 删除子仓库：`rm -r $prefix`

### 强制推送子仓库
> [How do I force a subtree push to overwrite remote changes? - Stack Overflow](https://stackoverflow.com/questions/33172857/how-do-i-force-a-subtree-push-to-overwrite-remote-changes)
**使用场景**：有时因为 rebase 等操作导致远程子仓库与本地不一致，而 `git subtree push` 并不支持 `--force` 选项，导致推送失败；

**命令**：
```sh
git push --force $name `git subtree split --prefix=$prefix --branch $name --rejoin`:$remote_branch
# --force 强制推送
# $name: 子仓库的远程地址别名，即`git remote add $name xxx.git` 中的 $name
# $remote_branch: 子仓库远程目标分支，一般为 master 或 main
```

### 注意事项：
* 不要对 git subtree split 产生的 commit 进行 rebase/merge 操作，会导致文件错乱！


## 修改 commit 的 author 信息
> 如何修改git commit的author信息 - 咸咸海风 - 博客园 | https://www.cnblogs.com/651434092qq/p/11015901.html


## 常用统计

### 统计 commit 次数

**总次数**
```sh
$ git log | grep '^commit ' | awk '{print $1}' | uniq -c | awk '{print $1}'
# 10
```

**每个人提交的次数，并排序**
> [git查看commit提交次数和代码量-CSDN博客](https://blog.csdn.net/cyf15238622067/article/details/82980782)
```sh
git log | grep "^Author: " | awk '{print $2}' | sort | uniq -c | sort -k1,1nr
# 10 aa
# 8 bb
```
