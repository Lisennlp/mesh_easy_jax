import ray
import time
import numpy as np
from queue import Queue

from mesh_transformer.util import head_print
from jax.experimental.multihost_utils import host_local_array_to_global_array, global_array_to_host_local_array
from jax.experimental import PartitionSpec as P


@ray.remote(resources={"tpu": 1})
class NetworkRunner(object):
    def __init__(self, mesh_shape, network_builder, version):
        self.mesh_shape = mesh_shape
        self.network_builder = network_builder
        self.version = int(version)

        self.input_q = Queue(maxsize=1)
        self.output_q = Queue(maxsize=1)

    def run(self):
        print(f"jax runtime initialization starting")
        import jax
        from jax.experimental.maps import thread_resources, ResourceEnv, Mesh
        import haiku as hk
        # jax.experimental.maps.EXPERIMENTAL_SPMD_LOWERING = True

        thread_resources.env = ResourceEnv(Mesh(np.empty((), dtype=object), ()), ())

        start = time.time()
        jax.devices()

        import warnings
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", category=ResourceWarning)

        if jax.host_id() == 0:
            warnings.filterwarnings("default")

        head_print(f"jax devices: {jax.device_count()}")
        head_print(f"jax runtime initialized in {time.time() - start:.06}s")
        devices = np.array(jax.devices()).reshape(self.mesh_shape)
        mesh = jax.experimental.maps.Mesh(devices, ('dp', 'mp'))
        with mesh:
            start = time.time()
            # model init
            network = self.network_builder()
            if self.version == 3:
                network.init_state()
            head_print(f"Initialized in {time.time() - start:.06}s")
            while True:
                operation, input = self.input_q.get()
                if operation in ["train", "eval"]:
                    input = host_local_array_to_global_array(input, mesh, P(None, 'dp'))
                    self.output_q.put(network.train(input, mode=operation))
                elif operation == "generate":
                    self.output_q.put(network.generate(*input))
                elif operation == "write_ckpt":
                    path, shard = input
                    network.write_ckpt(path, shard)
                    self.output_q.put(None)
                elif operation == "load_ckpt":
                    network.load_ckpt(input)
                    self.output_q.put(network.state["step"][0])
                elif operation == "get_params":
                    self.output_q.put(hk.data_structures.tree_size(network.state['params']))
                elif operation == "move_params":
                    # only needed for inference, otherwise first train step does this
                    local_shards = max(jax.local_device_count() // self.mesh_shape[1], 1)

                    # delete the optimizer states otherwise it OOMs for some reason
                    # TODO: use ShardedDeviceArray or something to get around this for bigger models
                    del network.state["opt_state"]
                    network.state = network.move_xmap(network.state, np.zeros(local_shards))
                    self.output_q.put(None)
                else:
                    raise Exception("Not implemented")

    def get_params(self):
        self.input_q.put(("get_params", None))
        return self.output_q.get()

    def train(self, sample):
        self.input_q.put(("train", sample))
        return self.output_q.get()

    def eval(self, sample):
        self.input_q.put(("eval", sample))
        return self.output_q.get()

    def generate(self, input):
        self.input_q.put(("generate", input))
        return self.output_q.get()

    def write_ckpt(self, path, shard):
        self.input_q.put(("write_ckpt", (path, shard)))
        return self.output_q.get()

    def load_ckpt(self, path):
        self.input_q.put(("load_ckpt", path))
        return self.output_q.get()

    def move_params(self):
        self.input_q.put(("move_params", None))
        return self.output_q.get()
