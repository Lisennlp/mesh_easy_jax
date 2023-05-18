1、optax的版本为0.9.0

2、如果提示：AttributeError: module 'jax' has no attribute 'tree_multimap'
将目标文件的tree_map进行替换::%s/jax\.tree_map/getattr\(jax, 'tree_multimap', jax\.tree_map\)/g

3、
