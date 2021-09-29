# 主目录
repo="$HOME/workspace/my/studies"
cd "$repo" || exit

# 执行文档测试
printf "=== Start Doc Test ===\n"
test_path="$repo/code/my"
out=$(python code/scripts/doctest_modules_recur.py "$test_path")
if [[ $out != 0 ]]; then
  echo "Not all file pass doctest!"
  exit
else
  echo "All file passed doctest."
fi
echo

commit_info="Init（清理历史commits）"

rm -rf .git
git init -b master
git config --local user.name imhuay
git config --local user.email imhuay@163.com
echo

printf "=== Start Push Main Repo ===\n"
git remote add origin "https://github.com/imhuay/studies.git"
git add -A
git commit -m "$commit_info"
git push --force --set-upstream origin master
echo


# 子仓库
sub_name="algorithm"
prefix="algorithm"  # path="algorithm"，不能使用 path 作为变量名

echo "=== Start Push $sub_name ==="
git remote add $sub_name "https://github.com/imhuay/algorithm.git"
sub_commit_id=$(git subtree split --prefix=$prefix --branch $sub_name --rejoin)
git push --force $sub_name "$sub_commit_id:master"
echo