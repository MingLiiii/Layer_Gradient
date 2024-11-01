import os
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default='try.jsonl')
    parser.add_argument("--model_name_or_path", type=str, default='meta-llama/Meta-Llama-3.1-8B')
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--start_idx_ratio", type=float, default=0)
    parser.add_argument("--end_idx_ratio", type=float, default=-1)
    parser.add_argument("--cache_dir", type=str, default='../cache')
    parser.add_argument("--run_instruct_version", action='store_true')
    args = parser.parse_args()
    return args


def cal_svd_vector_part_text(tokenizer, model, text, target_text, max_length, dict_temp):

    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    start_index = text.rfind(target_text)
    start_token = len(tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if len(param.shape) > 1:
            if (param.shape[0] == param.shape[1]) or ('self_attn' in name):
                param.requires_grad = True

    labels = input_ids.clone()

    labels[0, :start_token] = -100

    outputs = model(input_ids, labels=labels)

    loss = outputs.loss
    dict_temp['loss'] = loss.item()

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:

            # Calculate Matrix Mean, Max, Min
            M_mean = param.grad.mean()
            M_max = param.grad.max()
            M_min = param.grad.min()

            # Calculate Frobenius norm
            frobenius_norm = torch.linalg.norm(param.grad)

            # Do the SVD
            if len(param.grad.shape) > 1:
                _, S, _ = torch.linalg.svd(param.grad)
                S_sum = S.sum()
                S_max = S.max()
                S_min = S.min()

            dict_temp[name] = {}
            dict_temp[name]['M_mean'] = M_mean.item()
            dict_temp[name]['M_max'] = M_max.item()
            dict_temp[name]['M_min'] = M_min.item()

            dict_temp[name]['frobenius_norm'] = frobenius_norm.item()

            dict_temp[name]['S_sum'] = S_sum.item()
            dict_temp[name]['S_max'] = S_max.item()
            dict_temp[name]['S_min'] = S_min.item()

    model.zero_grad()

    return dict_temp


def filter_dicts(list1, list2):

    ids_in_list1 = {item['instruction'] for item in list1}

    filtered_dicts = [item for item in list2 if item['instruction'] not in ids_in_list1]

    return filtered_dicts


def main():

    args = parse_args()
    print(args)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", output_hidden_states=True, cache_dir=args.cache_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    model.eval()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    if args.start_idx_ratio > 1:
        start_idx = int(args.start_idx_ratio)
    else:
        start_idx = int(len(data) * args.start_idx_ratio)

    end_idx_ratio = args.end_idx_ratio if args.end_idx_ratio != -1 else 1
    if end_idx_ratio > 1:
        end_idx = int(end_idx_ratio)
    else:
        end_idx = int(len(data) * end_idx_ratio)

    sampled_data = data[start_idx:end_idx]

    dir_path = Path(args.save_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as file:
            pass  

    exsisting_jsonl_data = []
    with open(args.save_path, 'r') as jsonl_file:
        for line in jsonl_file:
            exsisting_jsonl_data.append(json.loads(line))

    sampled_data = filter_dicts(exsisting_jsonl_data, sampled_data)


    for i, data_i in tqdm(enumerate(sampled_data), total=len(sampled_data)):

        instruct_i = data_i['instruction']
        output_i = data_i['output']

        input_i = data_i['input'] if 'input' in data_i.keys() else ''
        if input_i != '':
            instruct_i = instruct_i + '\n' + input_i

        if args.run_instruct_version:
            if 'gemma' in args.model_name_or_path:
                messages = [
                    {"role": "user", "content": instruct_i}
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": instruct_i}
                ]
            instruct_i_it = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            whole_text = instruct_i_it+output_i
        else:
            whole_text = instruct_i+'\n'+output_i

        dict_temp = {}
        dict_temp['instruction'] = instruct_i
        dict_temp['output'] = output_i
        try:
            dict_temp = cal_svd_vector_part_text(tokenizer, model, whole_text, output_i, args.max_length, dict_temp)
        except:
            print(f"Error in {i}th data, skip it\n", dict_temp)
            continue

        with open(args.save_path, "a") as file:
            file.write(json.dumps(dict_temp) + '\n')

        pass


if __name__ == "__main__":
    main()