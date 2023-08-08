LLAMA_STANDARD_CONFIGS = {
    '7b': {
        'dim': 4096,
        'intermediate_size': 11008,
        'n_layers': 32,
        'n_heads': 32,
        'norm_eps': 1e-6,
    },
    '13b': {
        'dim': 5120,
        'intermediate_size': 13696, # baichuan
        'n_layers': 40,
        'n_heads': 40,
        'norm_eps': 1e-6,
    },
    '30b': {
        'dim': 6656,
        'intermediate_size': 17920,
        'n_layers': 60,
        'n_heads': 52,
        'norm_eps': 1e-6,
    },
    '65b': {
        'dim': 8192,
        'intermediate_size': 22016,
        'n_layers': 80,
        'n_heads': 64,
        'norm_eps': 1e-5,
    },
}

def match_keywords(string, positives, negatives):
    for positive in positives:
        if positive not in string:
            return False
    for negative in negatives:
        if negative in string:
            return False
    return True

def compute_intermediate_size(n):
    return int(math.ceil(n * 8 / 3) + 255) // 256 * 256


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(loaded, model_path, model_size):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    params = LLAMA_STANDARD_CONFIGS[model_size]

    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    # permute for sliced rotary
    def permute(w):
        """note: w must be torch type"""
        return w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)

    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        q = permute(loaded[f"transformer.h.{layer_i}.attention.wq.kernel"].transpose(1, 0))
        k = permute(loaded[f"transformer.h.{layer_i}.attention.wk.kernel"].transpose(1, 0))
        v =  loaded[f"transformer.h.{layer_i}.attention.wv.kernel"].transpose(1, 0)
        state_dict = {
            f"model.layers.{layer_i}.self_attn.W_pack.weight": torch.cat([q, k, v], dim=0),
            f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"transformer.h.{layer_i}.attention.wo.kernel"].transpose(1, 0),

            f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"transformer.h.{layer_i}.feed_forward.w1.kernel"].transpose(1, 0),
            f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"transformer.h.{layer_i}.feed_forward.w2.kernel"].transpose(1, 0),
            f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"transformer.h.{layer_i}.feed_forward.w3.kernel"].transpose(1, 0),

            f"model.layers.{layer_i}.input_layernorm.weight": loaded[f"transformer.h.{layer_i}.attention_norm.kernel"],
            f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[f"transformer.h.{layer_i}.ffn_norm.kernel"],

        }
        # state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    # Unsharded
    state_dict2 = {
        "model.embed_tokens.weight": loaded["transformer.wte.embedding"],
        "model.norm.weight": loaded["transformer.ln_f.kernel"],
        "lm_head.weight": loaded["lm_head.kernel"].transpose(1, 0),
    }
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict2, os.path.join(tmp_model_path, filename))
    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    state_dict2.update(state_dict)
    return state_dict2


def load_checkpoint(path, step=None):
    with jax.default_device(jax.devices("cpu")[0]):
        # orbax.checkpoint.Checkpointer可以不用tpu，但是orbax.checkpoint.AsyncCheckpointer
        item = {'params': orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())}
        mngr = orbax.checkpoint.CheckpointManager(path, item)
    if step is None:
        step = mngr.latest_step()
    w = mngr.restore(step)
    return w


def main(read_dir, save_dir, step, model_size):
    start = time.time()
    print(f'Start load checkpoint......')
    w = load_checkpoint(read_dir, step)
    loaded_w = {}
    for k, v in flatten_dict(w['params']['params']).items():
        k = '.'.join(k)
        loaded_w[k] = torch.from_numpy(v).to(torch.float16)
    print(f'Load checkpoint finished, take time: {time.time() - start}s......\n')
    
    print(f'Now start convert and write checkpoint...')
    start = time.time()
    write_model(loaded_w, model_path=save_dir, model_size=model_size)
    print(f'Convert and write checkpoint finished, take time: {time.time() - start}s...')


if __name__ == '__main__':
    # CPU only
    # pip install torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu
    # usage:
    # python convert_orbax_baichuan_to_hf.py --read_dir gs://llm_base_models/baichuan-13b/ --save_dir ./baichuan-13b --model_size 13b --step 8102
    parser = argparse.ArgumentParser(description='orbax to hf format script')

    parser.add_argument('--read_dir', type=str, help='Need to be converted model weight dir. it is a dir, stong recomand use local dir instead of cloud bucket.')
    parser.add_argument('--save_dir', type=str,  help='Save model weight file path, it is a dir.')
    parser.add_argument('--model_size', type=str, default='7b', choices=['7b', '13b', '30b', '65b'], help='model size')
    parser.add_argument('--step', type=int, default=None, help='load checkpoint step if None, load latest step checkpoint')

    args = parser.parse_args()

    model_size = args.model_size
    read_dir = args.read_dir
    save_dir = args.save_dir
    step = args.step

    print(f'read_dir: {args.read_dir}')
    print(f'save_dir: {args.save_dir}')
    print(f'model_size: {args.model_size}')

    main(read_dir, save_dir, step, model_size)
