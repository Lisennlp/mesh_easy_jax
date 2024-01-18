1、optax的版本为0.9.0

2、如果提示：AttributeError: module 'jax' has no attribute 'tree_multimap'
将目标文件的tree_map进行替换::%s/jax\.tree_map/getattr\(jax, 'tree_multimap', jax\.tree_map\)/g

3、



Pile v4-16-10  2.8B
TPU_NAME=llm-jax-v4-16-10
ZONE=us-central2-b
STEP=0
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd /home/lishengping/;sudo rm -r projects/*; cd projects/; git clone -b paxml https://github.com/Lisennlp/mesh_easy_jax.git"

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd projects/mesh_easy_jax; /home/lishengping/miniconda3/bin/pip install -r requirements.txt"


gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="killall main.py;/home/lishengping/miniconda3/bin/python /home/lishengping/projects/mesh_easy_jax/device_train.py --config configs/v4_16_test.json | tee test.log"