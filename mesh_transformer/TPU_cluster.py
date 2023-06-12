import itertools
import json
import time

import ray

from typing import Callable
import numpy as np

from mesh_transformer.train_actor import NetworkRunner
from google.cloud import storage
from smart_open import open
from func_timeout import func_set_timeout
import jax


class TPUCluster:
    @func_set_timeout(1800)
    def __init__(self,
                 mesh_shape,
                 node_count,
                 model: Callable,
                 version=1):
        assert ray.is_initialized()  # needs a valid ray cluster to start
        self.nodes = []
        self.node_count = node_count
        self.dp, self.mp = mesh_shape
        self.version = version

        start = time.time()

        for i in range(node_count):
            self.nodes.append(NetworkRunner.options(max_concurrency=2).remote(mesh_shape, model, version))
        runs = []
        for n in self.nodes:
            # 这里并不会执行run
            runs.append(n.run.remote())
        # 执行run函数
        # 不能先执行run
        # ray.get(runs)
        params = []
        for n in self.nodes:
            params.append(n.get_params.remote())
        #lsp: 在执行ray.get的时候，才会去执行run
        self.param_count = ray.get(params)[0]
        print(f"Ray actors created in {time.time() - start:.06}s")

    @func_set_timeout(1800)
    def train(self, data):
        masks = data['labels'] > 0
        input_ids = data['input_ids']
        data_chunks = np.array_split(input_ids, len(self.nodes), axis=1)
        mask_chunks = np.array_split(masks, len(self.nodes), axis=1)

        res = []
        for n, d, m in zip(self.nodes, data_chunks, mask_chunks):
            res.append(n.train.remote({
                "obs": d[:, :, :-1],
                "target": d[:, :, 1:],
                "masks": m[:, :, :-1],
            }))

        res = ray.get(res)

        loss = []
        acc = []

        for r in res:
            loss.append(r[0])
            acc.append(r[1])

        return np.array(loss).mean(), np.array(acc).mean()

    @func_set_timeout(1800)
    def eval(self, data):
        masks = data['labels'] > 0
        input_ids = data['input_ids']
        data_chunks = np.array_split(input_ids, len(self.nodes), axis=1)
        mask_chunks = np.array_split(masks, len(self.nodes), axis=1)

        res = []
        for n, d, m in zip(self.nodes, data_chunks, mask_chunks):
            res.append(n.eval.remote({
                "obs": d[:, :, :-1],
                "target": d[:, :, 1:],
                "masks": m[:, :, :-1],
            }))

        res = ray.get(res)

        loss = []
        acc = []

        for r in res:
            loss.append(r[0])
            acc.append(r[1])

        return np.array(loss).mean(), np.array(acc).mean()

    @func_set_timeout(1800)
    def generate(self, context, ctx_length, gen_len):
        context = np.array_split(context, len(self.nodes), axis=0)
        ctx_length = np.array_split(ctx_length, len(self.nodes), axis=0)

        res = []
        for n, ctx, l in zip(self.nodes, context, ctx_length):
            res.append(n.generate.remote((
                ctx,
                np.ones(len(ctx), dtype=np.uint32) * l,
                gen_len
            )))

        return np.concatenate([i[1][0][:, :, 0] for i in ray.get(res)], axis=0)

    @func_set_timeout(1800)
    def move(self):
        start = time.time()
        res = []
        for node in self.nodes:
            res.append(node.move_params.remote())
        ray.get(res)

        print(f"Moved weights to TPU in {time.time() - start:.06}s")

    @func_set_timeout(1800)
    def load(self, bucket, path):
        with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
            meta = json.load(f)

        ckpt_step = meta["checkpoints"][-1]

        # do replicated checkpoint reading
        start = time.time()
        res = []
        for node in self.nodes:
            res.append(node.load_ckpt.remote(f"gs://{bucket}/{path}/step_{ckpt_step}/"))

        # make sure they all read from the same checkpoint
        step = np.array(ray.get(res))
        assert (step[0] == step).all()
        step = int(step[0])

        print(f"Checkpoint@step{step} restored in {time.time() - start:.06}s")
        return step, meta["aux"][str(ckpt_step)]

    @func_set_timeout(1800)
    def save(self, step, bucket, path, aux=None, init=False, overwrite=False, keep_n=3, delete_old=True):
        assert path
        client = storage.Client()

        if aux is None:
            aux = {}

        # 删除所有历史checkpoint
        # if init:
        #     # check existing checkpoint folder does not exist, and delete it if it does
        #     for blob in client.list_blobs(bucket, prefix=f"{path}/"):
        #         assert overwrite
        #         # print(f"deleting {blob.name}")
        #         assert path in blob.name
        #         blob.delete()

        #     # create metadata file
        #     with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
        #         json.dump({
        #             "step": 0,
        #             "checkpoints": [],
        #             "aux": {}
        #         }, f)

        # do sharded checkpoint writing
        start = time.time()
        res = []

        if self.version == 1:
            for shard_id, node in zip(range(self.mp), itertools.cycle(self.nodes)):
                res.append(node.write_ckpt.remote(f"gs://{bucket}/{path}/step_{step}/", shard_id))
        elif self.version == 2:
            for node in self.nodes:
                res.append(node.write_ckpt.remote(f"gs://{bucket}/{path}/step_{step}", 0))
        elif self.version == 3:
            for shard_id, node in zip(range(self.mp), itertools.cycle(self.nodes)):
                res.append(node.write_ckpt.remote(f"gs://{bucket}/{path}/step_{step}/", shard_id))

        ray.get(res)
        print(f"Wrote checkpoint in {time.time() - start:.06}s")

        with open(f"gs://{bucket}/{path}/meta.json", "r") as f:
            meta = json.load(f)

        meta["step"] = step
        meta["checkpoints"].append(step)
        all_aux = meta.get("aux", {})

        while len(meta["checkpoints"]) > keep_n:
            ckpt_to_delete = meta["checkpoints"].pop(0)
            try:
                del all_aux[str(ckpt_to_delete)]
            except:
                print(f"failed to delete the aux state for {step}")

            if delete_old:
                print(f"deleting checkpoint {ckpt_to_delete}")
                if int(ckpt_to_delete) == 0:
                    continue
                for blob in client.list_blobs(bucket, prefix=f"{path}/step_{ckpt_to_delete}/"):
                    # print(f"deleting {blob.name}")
                    assert path in blob.name
                    blob.delete()
            else:
                print(f"keeping checkpoint {ckpt_to_delete}")

        all_aux[step] = aux
        meta["aux"] = all_aux

        with open(f"gs://{bucket}/{path}/meta.json", "w") as f:
            json.dump(meta, f)
