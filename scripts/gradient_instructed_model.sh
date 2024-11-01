python code_gradient/get_gradient_values.py \
    --data_path data/cot_500/aqua_train_no_cot.json \
    --save_path grads_exp/cot_500/llama3_8b_it_new/grads_aqua_train_no_cot.jsonl \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct  \
    --max_length 1024 \
    --run_instruct_version