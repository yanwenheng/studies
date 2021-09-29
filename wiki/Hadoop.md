Hadoop 常用命令
===

Index
---
<!-- TOC -->

- [参考](#参考)
- [常用](#常用)
    - [查看](#查看)
    - [下载文件](#下载文件)
    - [上传文件](#上传文件)

<!-- /TOC -->


### 参考
- Apache Hadoop – Overview | https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/FileSystemShell.html

### 常用

#### 查看
```shell
> hadoop fs -ls /user/hadoop/

# 
> hadoop fs -head pathname

```

#### 下载文件
```shell
> hadoop fs -get /user/hadoop/file output.txt

# 下载合并
> hadoop fs -getmerge /user/hadoop/* output.txt

# 效果同 getmerge
> hadoop fs -cat file1 file2 > output.txt
```

#### 上传文件
```shell
> hadoop fs -put localfile /user/hadoop/file.txt

# -f: 覆盖已有文件。
# -p: 保留访问和修改时间，所有权和权限。
> hadoop fs -put -f -p localfile1 localfile2 /user/hadoop/
```
- ``
- ``  



hadoop fs -cat /user/hadoop-aipnlp/bert_training_platform/work_space/dpsearch/predicts/5587/* /user/hadoop-aipnlp/bert_training_platform/work_space/dpsearch/predicts/5588/* /user/hadoop-aipnlp/bert_training_platform/work_space/dpsearch/predicts/5589/* /user/hadoop-aipnlp/bert_training_platform/work_space/dpsearch/predicts/5590/* /user/hadoop-aipnlp/bert_training_platform/work_space/dpsearch/predicts/5591/* > pred__a__b__branchname_ret.txt