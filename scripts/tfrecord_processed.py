import tensorflow as tf
import numpy as np
import time
import json
import sys


def yield_data(path):
    with open(path, 'r') as f:
        for line in f:
            line = json.loads(line)
            yield line

def shard(data, batch_size=None):  # XD
    return jax.tree_map(lambda x: x.numpy().reshape(batch_size + x.shape[1:]), data)  # mtj

# mesh-transformer-jax, https://www.tensorflow.org/tutorials/load_data/tfrecord
def _int64_feature(value): return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
# def _byte_feature(value): return tf.train.Feature(int64_list=tf.train.BytesList(value=value))


def write_tfrecords(dataset, fp):
    with tf.io.TFRecordWriter(fp) as writer:
        index = 0
        while True:
            example = next(dataset)
            if index % 100 == 0:
                print(f'processed: {index} time: {time.time() - start}')
            # 指定key，和value；key位数据字段名，value位feature类型
            feature = {
                "input_ids": _int64_feature(example['input_ids']),
                "attention_mask": _int64_feature(example['attention_mask']),
                "labels": _int64_feature(example['labels']),
                      }
            # 序列化， tf.train.Features(feature=feature)转化为feature
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            index += 1


file_path = sys.argv[1]
mode= sys.argv[2]
print(f'file_path: {file_path}')
start = time.time()
#wp = f'gs://jax_llm_data/small_dreamily_translation_general.{mode}.tfrecords'
wp = f'baichuan-data/dreamily_translation_general.{mode}.tfrecords'
print(f'save_path: {wp}')
test_data = yield_data(file_path)
write_tfrecords(test_data, wp)
end = time.time()
print(f'processed tfrecord data finished. take time: {end - start}s')