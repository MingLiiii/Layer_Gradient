import os
import json
import argparse
import numpy as np

import matplotlib.pyplot as plt

from exp_utils import get_curve_dict, draw_figure

MODEL_DICT = {
    "gemma2_2b": "gemma-2-2b",
    "gemma2_2b_it_new": "gemma-2-2b-it",
    "llama3_8b": "Llama-3.1-8B",
    "llama3_8b_it_new": "Llama-3.1-8B-Instruct",
    "gemma_2_9b": "gemma-2-9b",
    "gemma_2_9b_it_new": "gemma-2-9b-it",
    "qwen2_1_5b": "Qwen2-1.5B",
    "qwen2_1_5b_it_new": "Qwen2-1.5B-Instruct",
    "llama2_7b": "Llama-2-7b-hf",
    "llama2_7b_it_new": "Llama-2-7b-chat-hf",
}

def read_jsonl_to_json(jsonl_path):
    json_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            json_data.append(json.loads(line.strip()))
    return json_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama3_8b')
    parser.add_argument("--task_type", type=str, default='wiki_popularity')
    # lighteval_math, wiki_popularity_wrong_answer_shuffle
    parser.add_argument("--data_name", type=str, default='grads')
    parser.add_argument("--attribute_list", type=list, default=['S_sum', 'S_ratio'])
    parser.add_argument("--table_label", type=str, default='wiki_')
    parser.add_argument("--include_ratio", type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    jsonl_file_path_1 = os.path.join('grads_exp',args.task_type,args.model_name,args.data_name+'_popular_wikipedia_100.jsonl')
    jsonl_file_path_2 = os.path.join('grads_exp',args.task_type,args.model_name,args.data_name+'_popular_wikipedia_500.jsonl')
    jsonl_file_path_3 = os.path.join('grads_exp',args.task_type,args.model_name,args.data_name+'_popular_wikipedia_1000.jsonl')
    jsonl_file_path_4 = os.path.join('grads_exp',args.task_type,args.model_name,args.data_name+'_long_tail_wikipedia.jsonl')

    json_data_1 = read_jsonl_to_json(jsonl_file_path_1)
    json_data_2 = read_jsonl_to_json(jsonl_file_path_2)
    json_data_3 = read_jsonl_to_json(jsonl_file_path_3)
    json_data_4 = read_jsonl_to_json(jsonl_file_path_4)

    json_data_list = [json_data_1, json_data_2, json_data_3, json_data_4]

    # Get curve dict with MAD
    curve_dict = get_curve_dict(args, json_data_list)

    # Get label
    args.table_label = args.table_label+args.task_type+'_'+args.model_name+'_'+args.data_name

    # Get real dataset name
    real_name = 'Wiki'
    real_model_name = MODEL_DICT[args.model_name]

    save_dir = f'{args.model_name}_results'
    os.makedirs(save_dir, exist_ok=True)

    # Draw figure
    figure_save_path = args.table_label + '.png'
    figure_save_path = os.path.join(save_dir, figure_save_path)

    text_list = ['Length 100', 'Length 500', 'Length 1000', 'Unpopular']
    draw_figure(args, curve_dict, figure_save_path, text_list, 30)

    pass