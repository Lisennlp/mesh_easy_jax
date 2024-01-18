#!/bin/bash

REMOTE_PROJECTS_DIR=/home/lishengping/projects
# 获取命令行参数
file_name=$3
direction=$2
tpu_suffix=$1

echo "file_name: $file_name"
echo "direction: $direction"
echo "tpu_name: llm-jax-$tpu_suffix"

# # 检查命令行参数是否存在
# if [ $# -ne 2 ]; then
#   echo "Please provide file name and ftp direction"
#   exit 1
# fi

abs_path=$(readlink -f "$file_name")
# path='/home/lisen/projects/paxml/paxml/a.txt'
names=("paxml" "praxis" "mesh_easy_jax")

echo "Abs path: $abs_path"
# 将路径分割为目录组成的数组
IFS='/' read -ra path_parts <<< "$abs_path"
# 遍历数组，查找匹配的目录名
matched_paths=()
for name in "${names[@]}"; do
    for ((i=0; i<${#path_parts[@]}; i++)); do
        if [[ "${path_parts[i]}" == "$name" ]]; then
            matched_paths+=("${path_parts[i]}")
            for ((j=i+1; j<${#path_parts[@]}; j++)); do
                matched_paths+=("${path_parts[j]}")
            done
            break
        fi
    done
done

if [ ${#matched_paths[@]} -gt 0 ]; then
    MATCH_PATH=$(echo "${matched_paths[*]}" | tr ' ' '/')
    echo "Matched path: $MATCH_PATH"
fi

if [[ $tpu_suffix == *v3* ]]; then
  zone="us-east1-d"
elif [[ $tpu_suffix == *v4* ]]; then
  zone="us-central2-b"
else
  zone="us-west4-a"
fi
echo "Zone: $zone"

remote_path=$REMOTE_PROJECTS_DIR/$MATCH_PATH
echo "Romote path: $remote_path"
# 检查传输方向
if [ "$direction" -eq 0 ]; then
  # 从A传至B
  gcloud compute tpus tpu-vm scp $file_name  llm-jax-${tpu_suffix}:$remote_path --worker all --zone $zone
  echo "File name <${file_name}> have been ftped to <$remote_path> successfully"
elif [ "$direction" -eq 1 ]; then
  # 从B传至A
  gcloud compute tpus tpu-vm scp llm-jax-${tpu_suffix}:$remote_path ./ --worker all --zone $zone
  echo "File <${file_name}> have been ftped to <./> successfully"
else
  echo "无效的传输方向。传输方向应为0或1。"
fi
