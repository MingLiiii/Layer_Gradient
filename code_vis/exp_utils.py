import os
import json
import argparse
import numpy as np

import matplotlib.pyplot as plt

def relative_difference(list1, list2):

    if len(list1) != len(list2):
        raise ValueError("The two lists must have the same length.")

    arr1 = np.array(list1, dtype=float)
    arr2 = np.array(list2, dtype=float)

    arr1[arr1 == 0] = 1e-10
    rd = np.abs(arr1 - arr2) / np.abs(arr1)
    avg_rd = np.mean(rd)

    top_5_indices = np.argsort(rd)[-5:][::-1] 

    return avg_rd, top_5_indices.tolist()

def mean_absolute_difference(values):
    absolute_differences = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    mad_value = sum(absolute_differences) / len(absolute_differences)
    return mad_value

def cal_MAD_3_devides(value_list):
    mad_all = mean_absolute_difference(value_list)

    if len(value_list) == 28:
        mad_1 = mean_absolute_difference(value_list[:7])
        mad_2 = mean_absolute_difference(value_list[7:21])
        mad_3 = mean_absolute_difference(value_list[21:])
        return mad_1, mad_2, mad_3, mad_all

    if len(value_list) == 26:
        mad_1 = mean_absolute_difference(value_list[:7])
        mad_2 = mean_absolute_difference(value_list[7:20])
        mad_3 = mean_absolute_difference(value_list[20:])
        return mad_1, mad_2, mad_3, mad_all

    if len(value_list) == 32:
        mad_1 = mean_absolute_difference(value_list[:10])
        mad_2 = mean_absolute_difference(value_list[10:20])
        mad_3 = mean_absolute_difference(value_list[20:])
        return mad_1, mad_2, mad_3, mad_all

    if len(value_list) == 42:
        mad_1 = mean_absolute_difference(value_list[:12])
        mad_2 = mean_absolute_difference(value_list[12:30])
        mad_3 = mean_absolute_difference(value_list[30:])
        return mad_1, mad_2, mad_3, mad_all

def parse_qkvo_each(args, data_each, attribute):

    q_list, k_list , v_list, o_list = [], [], [], []

    if ('llama2_7b' in args.model_name) or ('llama3_8b' in args.model_name):
        layer_num = 32
    if ('gemma_2b' in args.model_name) :
        layer_num = 18
    if ('gemma2_2b' in args.model_name) :
        layer_num = 26
    if ('gemma_2_9b' in args.model_name) :
        layer_num = 42
    if ('qwen2_1_5b' in args.model_name) :
        layer_num = 28

    for layer_i in range(layer_num):
        q_layer_name = 'model.layers.'+str(layer_i)+'.self_attn.q_proj.weight'
        k_layer_name = 'model.layers.'+str(layer_i)+'.self_attn.k_proj.weight'
        v_layer_name = 'model.layers.'+str(layer_i)+'.self_attn.v_proj.weight'
        o_layer_name = 'model.layers.'+str(layer_i)+'.self_attn.o_proj.weight'

        if attribute == 'S_ratio':
            try:
                q_list.append(data_each[q_layer_name]['S_max']/data_each[q_layer_name]['S_sum'])
                k_list.append(data_each[k_layer_name]['S_max']/data_each[k_layer_name]['S_sum'])
                v_list.append(data_each[v_layer_name]['S_max']/data_each[v_layer_name]['S_sum'])
                o_list.append(data_each[o_layer_name]['S_max']/data_each[o_layer_name]['S_sum'])
            except ZeroDivisionError:
                q_list.append(0)
                k_list.append(0)
                v_list.append(0)
                o_list.append(0)

        elif attribute == 'S_sum_regularized':
            try:
                q_list.append(data_each[q_layer_name]['S_sum']/data_each['loss'])
                k_list.append(data_each[k_layer_name]['S_sum']/data_each['loss'])
                v_list.append(data_each[v_layer_name]['S_sum']/data_each['loss'])
                o_list.append(data_each[o_layer_name]['S_sum']/data_each['loss'])
            except ZeroDivisionError:
                q_list.append(0)
                k_list.append(0)
                v_list.append(0)
                o_list.append(0)

        else:
            q_list.append(data_each[q_layer_name][attribute])
            k_list.append(data_each[k_layer_name][attribute])
            v_list.append(data_each[v_layer_name][attribute])
            o_list.append(data_each[o_layer_name][attribute])

    
    return q_list, k_list , v_list, o_list


def parse_qkvo_all(args, data_all, attribute):

    q_list_list, k_list_list , v_list_list, o_list_list = [], [], [], []

    for i, data_i in enumerate(data_all):

        q_list, k_list , v_list, o_list = parse_qkvo_each(args, data_i, attribute)
        q_list_list.append(q_list)
        k_list_list.append(k_list)
        v_list_list.append(v_list)
        o_list_list.append(o_list)

    q_list_list = np.array(q_list_list)
    k_list_list = np.array(k_list_list)
    v_list_list = np.array(v_list_list)
    o_list_list = np.array(o_list_list)

    q_list_list_clean = q_list_list[~np.isnan(q_list_list).any(axis=1)]
    k_list_list_clean = k_list_list[~np.isnan(k_list_list).any(axis=1)]
    v_list_list_clean = v_list_list[~np.isnan(v_list_list).any(axis=1)]
    o_list_list_clean = o_list_list[~np.isnan(o_list_list).any(axis=1)]

    q_list_list_mean_list = q_list_list_clean.mean(0).tolist()
    k_list_list_mean_list = k_list_list_clean.mean(0).tolist()
    v_list_list_mean_list = v_list_list_clean.mean(0).tolist()
    o_list_list_mean_list = o_list_list_clean.mean(0).tolist()

    return q_list_list_mean_list, k_list_list_mean_list, v_list_list_mean_list, o_list_list_mean_list


def get_curve_dict(args, json_data_list):

    dict_temp = {}
    for i, json_data_i in enumerate(json_data_list):
        dict_temp[i] = {}
        for attribute in args.attribute_list:
            dict_temp[i][attribute] = {}

            q_list, k_list, v_list, o_list = parse_qkvo_all(args, json_data_i, attribute)
            dict_temp[i][attribute]['Q'] = {}
            dict_temp[i][attribute]['K'] = {}
            dict_temp[i][attribute]['V'] = {}
            dict_temp[i][attribute]['O'] = {}

            dict_temp[i][attribute]['Q']['curve'] = q_list
            dict_temp[i][attribute]['K']['curve'] = k_list
            dict_temp[i][attribute]['V']['curve'] = v_list
            dict_temp[i][attribute]['O']['curve'] = o_list

            dict_temp[i][attribute]['Q']['MAD'] = cal_MAD_3_devides(q_list)
            dict_temp[i][attribute]['K']['MAD'] = cal_MAD_3_devides(k_list)
            dict_temp[i][attribute]['V']['MAD'] = cal_MAD_3_devides(v_list)
            dict_temp[i][attribute]['O']['MAD'] = cal_MAD_3_devides(o_list)

    return dict_temp

def draw_figure(args, curve_dict, save_path, text_list, y_s_max_value=20):

    num_rows = 3 if args.include_ratio else 1
    num_cols = len(text_list)
    fig_hight = 10 if args.include_ratio else 5

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*5, fig_hight)) 
    y_r_max_value_1 = 0.6
    y_r_max_value_2 = 0.6

    if ('llama2_7b' in args.model_name) or ('llama3_8b' in args.model_name):
        layer_num = 32
    if ('gemma_2b' in args.model_name) :
        layer_num = 18
    if ('gemma2_2b' in args.model_name) :
        layer_num = 26
    if ('gemma_2_9b' in args.model_name) :
        layer_num = 42
    if ('qwen2_1_5b' in args.model_name) :
        layer_num = 28

    x = range(layer_num)

    for i in range(num_cols):

        if args.include_ratio:
            axs[0, i].plot(x, curve_dict[i]['S_sum']['Q']['curve'], c='r', label='Proj_Q')
            axs[0, i].plot(x, curve_dict[i]['S_sum']['K']['curve'], c='g', label='Proj_K')
            axs[0, i].plot(x, curve_dict[i]['S_sum']['V']['curve'], c='b', label='Proj_V')
            axs[0, i].plot(x, curve_dict[i]['S_sum']['O']['curve'], c='gray', label='Proj_O')
            axs[0, i].set_title(f'{text_list[i]} | Nuclaer Norm')
            axs[0, i].legend()
            axs[0, i].grid(True, color='lightgray', linestyle='--', linewidth=0.5, alpha=1)
            if 'cot' in args.task_type and i == 0:
                axs[0, i].set_ylim(0, 100)
            else:
                axs[0, i].set_ylim(0, y_s_max_value)

            axs[1, i].plot(x, curve_dict[i]['S_ratio']['Q']['curve'], c='r', label='Proj_Q')
            axs[1, i].plot(x, curve_dict[i]['S_ratio']['K']['curve'], c='g', label='Proj_K')
            axs[1, i].set_title(f'{text_list[i]} | Ratio (Q, K)')
            axs[1, i].legend()
            axs[1, i].grid(True, color='lightgray', linestyle='--', linewidth=0.5, alpha=1)

            axs[2, i].plot(x, curve_dict[i]['S_ratio']['V']['curve'], c='b', label='Proj_V')
            axs[2, i].plot(x, curve_dict[i]['S_ratio']['O']['curve'], c='gray', label='Proj_O')
            axs[2, i].set_title(f'{text_list[i]} | Ratio (V, O)')
            axs[2, i].legend()
            axs[2, i].grid(True, color='lightgray', linestyle='--', linewidth=0.5, alpha=1)

            axs[1, i].set_ylim(0, y_r_max_value_1)
            axs[2, i].set_ylim(0, y_r_max_value_2)

        else:
            axs[i].plot(x, curve_dict[i]['S_sum']['Q']['curve'], c='r', label='Proj_Q')
            axs[i].plot(x, curve_dict[i]['S_sum']['K']['curve'], c='g', label='Proj_K')
            axs[i].plot(x, curve_dict[i]['S_sum']['V']['curve'], c='b', label='Proj_V')
            axs[i].plot(x, curve_dict[i]['S_sum']['O']['curve'], c='gray', label='Proj_O')
            axs[i].set_title(f'{text_list[i]} | Nuclaer Norm')
            axs[i].legend()
            axs[i].grid(True, color='lightgray', linestyle='--', linewidth=0.5, alpha=1)

            if 'cot' in args.task_type and i == 0:
                axs[i].set_ylim(0, 100)
            else:
                axs[i].set_ylim(0, y_s_max_value)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
