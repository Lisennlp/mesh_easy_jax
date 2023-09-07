from datasets import load_dataset
import numpy as np
import mlxu
import os
import re
import time
import random
from typing import List
from collections import defaultdict
import json
from langdetect import detect


def lan_and_turns_counter(data, count_dict):
    human = data['conversations'][0]['human']
    assistant = data['conversations'][1]['assistant']
    try:
        language = detect(human)
        if 'zh' in language:
            count_dict['zh'] += 1
        elif 'en' in language:
            count_dict['en'] += 1
        else:
            count_dict['others'] += 1
    except Exception as error:
        count_dict['others'] += 1
        print(f'error: {error}, human: {human}')
    if len(data['conversations']) > 2:
        count_dict['multi'] += 1
    else:
        count_dict['single'] += 1


def replace_text(text):
    text = text.replace('\u3000', ' ').strip()
    text = text.replace('\\n', '\n')
    return text


# filter_dataset = dataset['train'].filter(lambda x: 'flan' in x['id'])
def key_map_fn(data, key_map):
    data = data.rename_columns(key_map)
    return data


def write_metadata(count_dict, save_path):
    count_dict['sum'] = count_dict['zh'] + count_dict['en']
    count_dict = json.dumps(count_dict, ensure_ascii=False)
    metadata_path = save_path.split('.')[0] + '.metadata'
    with mlxu.open_file(metadata_path, 'w') as fin:
        fin.write(count_dict)


def extract_and_save_data_one(save_path, split_data, indexes, key_map:List=None):
    """format: {instruction/kind: ..., input:..., output:...}"""
    start = time.time()
    if key_map is not None:
        split_data = key_map_fn(split_data, key_map)
    datas = []
    lan_turns_count_dict = defaultdict(int)
    with mlxu.open_file(save_path, 'w') as fin:
        for i, index in enumerate(indexes):
            data = {'id': f'{name}_{index}', 'conversations': []}
            instruction = ''
            if 'instruction' in split_data.column_names:
                instruction = split_data['instruction'][index].as_py().strip()
            input = split_data['input'][index].as_py().strip()
            output = split_data['output'][index].as_py().strip()
            if not output: continue
            if output[0] in ':：':
                output = output[1:]
            if not output: continue
            if '[HM]' in instruction:
                instruction += '\n'
                hm = re.findall('\[HM\]\:(.*?)\n', instruction)
                ai = re.findall('\[AI\]\:(.*?)\n', instruction)
                assert len(hm) - 1 == len(ai), print(hm, ai)
                for h, a in zip(hm[:-1], ai):
                    data['conversations'].append({'human': replace_text(h)})
                    data['conversations'].append({'assistant': replace_text(h)})
                cat_input = hm[-1] + '\n' + input
                data['conversations'].append({'human': replace_text(cat_input)})
            else:
                human = instruction + '\n\n' + input
                human = human.strip() # input和instruction有可能为空的情况
                data['conversations'].append({'human': replace_text(human)})
                
            data['conversations'].append({'assistant': replace_text(output)})
            datas.append(data)
            lan_and_turns_counter(data, lan_turns_count_dict)
            fin.write(f'{data}\n')
            if i % 10000 == 0:
                print(f'processing: {i}/{len(indexes)} take time: {(time.time() - start):03f}...')
                print(f'lan_turns_count_dict:\n{lan_turns_count_dict}')
                print(f'sample_{i}: {data}\n\n')
    write_metadata(lan_turns_count_dict, save_path)
    return datas, lan_turns_count_dict


def extract_and_save_data_two(save_path, split_data, indexes, key_map=None):
    """format: {..., conversations: ...}"""
    start = time.time()
    if key_map is not None:
        split_data = key_map_fn(split_data, key_map)
    datas = []
    lan_turns_count_dict = defaultdict(int)
    with mlxu.open_file(save_path, 'w') as fin:
        for i, index in enumerate(indexes):
            flag = 0
            data = {'id': f'{name}_{index}', 'conversations': []}
            conversations = split_data['conversation'][index].as_py()
            for conversation in conversations:
                human = conversation['human'].strip()
                assistant = conversation['assistant'].strip()
                if not human or not assistant: 
                    flag = 1
                    break
                data['conversations'].append({'human': replace_text(human)})
                data['conversations'].append({'assistant': replace_text(assistant)})
            if flag: continue
            datas.append(data)
            fin.write(f'{data}\n')
            lan_and_turns_counter(data, lan_turns_count_dict)
            if i % 10000 == 0:
                print(f'processing: {i}/{len(indexes)}, take time: {(time.time() - start):03f}...')
                print(f'lan_turns_count_dict:\n{lan_turns_count_dict}')
                print(f'sample_{i}: {data}\n\n')
    write_metadata(lan_turns_count_dict, save_path)
    return datas, lan_turns_count_dict


def extract_and_save_data_three(save_path, split_data, indexes, key_map=None):
    """format: {message: ['role': user/assistant, 'content': ...]}"""
    start = time.time()
    if key_map is not None:
        split_data = key_map_fn(split_data, key_map)
    datas = []
    lan_turns_count_dict = defaultdict(int)
    with mlxu.open_file(save_path, 'w') as fin:
        for i, index in enumerate(indexes):
            data = {'id': f'{name}_{index}', 'conversations': []}
            conversations = split_data['messages'][index].as_py()
            for j, conversation in enumerate(conversations):
                role = conversation['role']
                if j == 0 and role != 'user': break
                if role == 'user': role = 'human'
                content = conversation['content'].strip()
                if not content: break #  when content is null, throw away
                data['conversations'].append({role: replace_text(content)})
            datas.append(data)
            fin.write(f'{data}\n')
            lan_and_turns_counter(data, lan_turns_count_dict)
            if i % 10000 == 0:
                print(f'processing: {i}/{len(indexes)}, take time: {(time.time() - start):03f}...')
                print(f'lan_turns_count_dict:\n{lan_turns_count_dict}')
                print(f'sample_{i}: {data}\n\n')
    write_metadata(lan_turns_count_dict, save_path)
    return datas, lan_turns_count_dict


def extract_and_save_data_OpenOrca(save_path, split_data, indexes, key_map:List=None):
    """400万+ OpenOrca: {'id': flan/cot/t0..., 'question', 'response'}"""
    start = time.time()
    if key_map is not None:
        split_data = key_map_fn(split_data, key_map)
    datas = []
    lan_turns_count_dict = defaultdict(int)
    with mlxu.open_file(save_path, 'w') as fin:
        for i, index in enumerate(indexes):
            id = split_data['id'][index].as_py()
            data = {'id': f'{name}_{id}', 'conversations': []}
            if 'cot' not in id and 'flan' not in id:
                continue
            randint = random.randint(0, 10)
            if 'flan' in id and randint != 0:
                continue
            question = split_data['question'][index].as_py()
            response = split_data['response'][index].as_py()
            if not question or not response: 
                break
            
            data['conversations'].append({'human': replace_text(question)})
            data['conversations'].append({'assistant': replace_text(response)})
            datas.append(data)
            fin.write(f'{data}\n')
            lan_and_turns_counter(data, lan_turns_count_dict)
            if i % 10000 == 0:
                print(f'processing: {i}/{len(indexes)}, take time: {(time.time() - start):03f}...')
                print(f'lan_turns_count_dict:\n{lan_turns_count_dict}')
                print(f'sample_{i}: {data}\n\n')
    write_metadata(lan_turns_count_dict, save_path)
    return datas, lan_turns_count_dict
                
    
def processed_main(dataset_obj, name, ratio=1.0):
    for split in dataset_obj: 
        data = dataset_obj.data[split]
        size = data.shape[0]
        ratio_size = int(size * ratio)
        indexes = random.sample(range(size), ratio_size)
        save_path = os.path.join(save_dir, f'{name}.{split}.jsonl')
        print(f'save_path: {save_path}')
        key_map = None
        if name.lower() in ['alpaca-cot', 'coig', 'firefly-train-1.1m', 'flan_v2_cot_fs']:
            if 'firefly' in name.lower():
                key_map = ['kind', 'input', 'output']
            elif 'flan_v2_cot_fs' in name.lower():
                key_map=['input', 'output', 'task']
            datas, lan_turns_count_dict = extract_and_save_data_one(save_path, data, indexes, key_map=key_map)
        elif name.lower() in ['wizardlm_evol_instruct_v2_143k', 'moss-003-sft-data', 'ultrachat', 'sharegpt-chinese-english-90k']:
            datas, lan_turns_count_dict = extract_and_save_data_two(save_path, data, indexes)
        elif name.lower() in ['oasst1']:
            datas, lan_turns_count_dict = extract_and_save_data_three(save_path, data, indexes)
        elif name.lower() in ['openorca']:
            datas, lan_turns_count_dict = extract_and_save_data_OpenOrca(save_path, data, indexes)
        return datas, lan_turns_count_dict


if __name__ == '__main__':
    data_configs = {
    # CoT en: 74771, zh: 74771
    'Alpaca-CoT': {'data_files': ['CoT_data.json', 'CoT_Chinese_data.json'], 'data_dir': 'Chain-of-Thought', 'path': 'QingyiSi/Alpaca-CoT', 'ratio': 1.0},
    # 365928 multi lan and single
    'COIG': {'data_files': None, 'data_dir': 'COIG', 'path': 'QingyiSi/Alpaca-CoT', 'ratio': 0.3},
    # en: 143k and single
    'WizardLM_evol_instruct_V2_143k': {'data_files':None, 'data_dir': None, 'path': 'YeungNLP/WizardLM_evol_instruct_V2_143k', 'ratio': 0.3},
    # multi lan and multi turn：1074551
    'moss-003-sft-data': {'data_files':None, 'data_dir': None, 'path': 'YeungNLP/moss-003-sft-data', 'ratio': 0.2},
    # multi lan and single -> 1649399
    'firefly-train-1.1M': {'data_files':None, 'data_dir': None, 'path': 'YeungNLP/firefly-train-1.1M', 'ratio': 0.1},
    # 1468338
    'ultrachat': {'data_files': None, 'data_dir': None, 'path': 'YeungNLP/ultrachat', 'ratio': 0.1},
    # multi lan and  40k
    'oasst1': {'data_files': None, 'data_dir': 'data', 'path': 'ybelkada/oasst1', 'ratio': 1.0},
    # 挑id类型为：cot，flan，.... 英文cot：141695   总： 4233923
    'OpenOrca': {'data_files': None, 'data_dir': None, 'path': 'Open-Orca/OpenOrca', 'ratio': 0.6},
    # 155506
    'ShareGPT-Chinese-English-90k': {'data_files': ['common_en_70k.jsonl', 'common_zh_70k.jsonl'], 'data_dir': 'sharegpt_jsonl', 'path': 'shareAI/ShareGPT-Chinese-English-90k', 'ratio': 0.5},
    # 挑fewshot + cot数据 373681
    'flan_v2_cot_fs': {'data_files': ['cot_fs_noopt_train.jsonl.gz', 'cot_fs_opt_train.jsonl.gz', 'cot_zs_noopt_train.jsonl.gz', 'cot_zs_opt_train.jsonl.gz'], 'data_dir': None, 'path': 'SirNeural/flan_v2', 'ratio': 0.3}
    }
    config_keys = list(data_configs.keys())
    print(f'all datasets: {config_keys}')
    for name, value in data_configs.items():
        # if 'fire' in name or 'coig' in name: continue
        # name = config_keys[4]
        config = data_configs[name]
        print(f'Start processed dataset: {name}, Config is\n: {config}')
        kwargs = {k: v for k, v in config.items() if k in ['data_dir', 'path', 'data_files']}
        dataset = load_dataset(**kwargs, cache_dir='~/general_v2_cache')
        print(dataset)
        save_dir = 'gs://jax_llm_data/general/23.08.15'
        datas, lan_turns_count_dict = processed_main(dataset, name, ratio=config['ratio'])
        print(f'Dataset: {name} processed finished....\n\n')
