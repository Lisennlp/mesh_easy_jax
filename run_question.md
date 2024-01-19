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


gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="killall main.py;cd projects/mesh_easy_jax; /home/lishengping/miniconda3/bin/python device_train.py --config configs/v4_16_test.json | tee test.log"



export PROJECT_ID=colorful-aia
export ACCELERATOR_TYPE=v5litepod-16
export ZONE=us-west4-a
export RUNTIME_VERSION=v2-alpha-tpuv5-lite
export SERVICE_ACCOUNT=495921979093-compute@developer.gserviceaccount.com
export TPU_NAME=llm-jax-v5-16-10
export QUEUED_RESOURCE_ID=llm-jax-v5-16-10
export QUOTA_TYPE=best-effort

gcloud alpha compute tpus queued-resources create ${QUEUED_RESOURCE_ID} \
  --node-id ${TPU_NAME} \
  --project ${PROJECT_ID} \
  --zone ${ZONE} \
  --accelerator-type ${ACCELERATOR_TYPE} \
  --runtime-version ${RUNTIME_VERSION} \
  --service-account ${SERVICE_ACCOUNT} \
  --${QUOTA_TYPE}


gcloud container node-pools create llm-jax-v5-16-10 \
--cluster=tpu-cluster \
--machine-type=ct5lp-hightpu-1t \
--num-nodes=2 \
--location=us-west4-a \
--${QUOTA_TYPE}

gcloud container node-pools create llm-jax-v5-16-10 \
--cluster=tpu-cluster \
--machine-type=ct5lp-hightpu-1t \
--num-nodes=2 \
--location=us-west4-a
