#!/usr/bin/env bash

repo="$HOME/workspace/my/studies"
cd "$repo" || exit

# 执行文档测试
printf "=== Start Doc Test ===\n"
test_path="$repo/code/my"
out=$(python code/scripts/doctest_modules_recur.py "$test_path")
if [[ $out != 0 ]]; then
  echo "Not all file pass doctest($test_path)"
  exit
else
  echo "All file passed doctest."
fi
echo

# 生成 README.md
printf "=== Start generating README.md ===\n"
code_path="$repo/code"
out=$(python code/scripts/generate_readme_examples.py \
      --module_path "$code_path" \
      --out "$repo/code/README.md")
echo "$out"
if [[ $out = 'DIFF' ]]; then
  git add "README.md" "$repo/code/README.md"
  git commit -m '[U] Auto-README.md'
fi
echo

# 主仓库
printf "=== Start Push Main Repo ===\n"
git push
echo

# 统计 commit 次数
#num_commits=$(git log | grep '^commit ' | awk '{print $1}' | uniq -c | awk '{print $1}')
#split_feq=20  # 每提交 20 次再 split 一次
#split_flag=$((num_commits % split_feq))

# 子仓库
prefix="algorithm"
name="algorithm"
echo "=== Start Push $name ==="
git subtree split --prefix=$prefix --branch $name --rejoin
git subtree push --prefix=$prefix $name master --squash
echo

#prefix="code"
#name="my_lab"
#echo "=== Start Push $name ==="
#git subtree split --prefix=$prefix --branch $name --rejoin
#git subtree push --prefix=$prefix $name master --squash
#echo
#
#prefix="code/my"
#name="my"
#echo "=== Start Push $name ==="
#git subtree split --prefix=$prefix --branch $name --rejoin
#git subtree push --prefix=$prefix $name master --squash
#echo


#====================== history

# git subtree add --prefix=code/keras_demo/keras_model/bert_by_keras bert_by_keras master --squash
# git subtree add --prefix=code/keras_demo keras_demo master --squash

# 使用 submodule 代替 subtree
# git subtree push --prefix=code/keras_demo/keras_model/bert_by_keras bert_by_keras master
# git subtree push --prefix=code/keras_demo keras_demo master

# 获取仓库父目录
#pwd=$(pwd)

# 先更新子仓库
#printf "=== First: Update submodule ===\n"

# 1.
#sub_repo="bert_by_keras"
#echo "____ Start update $sub_repo"
#cd "$pwd/code/my_models/$sub_repo" || exit
#ret=$(git pull origin master)
#if [[ $ret =~ "Already up to date" ]]; then
#  echo "$sub_repo is already up to date."
#else
#  cd "$pwd" || exit
#  git add "$pwd/code/my_models/$sub_repo"
#  git commit -m "U $sub_repo"
#fi

# 更新父仓库
#cd "$pwd" || exit
#printf "\n=== Final: Push father repository ===\n"
#git push
