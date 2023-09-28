#!/usr/bin/env bash

gpu_num=1

data_dir=
test_subset=(test)

exp_name=
if [ "$#" -eq 1 ]; then
    exp_name=$1
fi

sacrebleu=0
n_average=5
beam_size=4
len_penalty=0.6
max_tokens=4000
batch_size=1
infer_debug=0
dec_model=checkpoint_best.pt

cmd="./run.sh
    --stage 2
    --stop_stage 2
    --gpu_num ${gpu_num}
    --exp_name ${exp_name}
    --sacrebleu ${sacrebleu}
    --n_average ${n_average}
    --beam_size ${beam_size}
    --len_penalty ${len_penalty}
    --batch_size ${batch_size}
    --max_tokens ${max_tokens}
    --dec_model ${dec_model}
    --infer_debug ${infer_debug}
    "

if [[ -n ${data_dir} ]]; then
    cmd="$cmd --data_dir ${data_dir}"
fi
if [[ ${#test_subset[@]} -ne 0 ]]; then
    subsets=$(echo ${test_subset[*]} | sed 's/ /,/g')
    cmd="$cmd --test_subset ${subsets}"
fi

echo $cmd
eval $cmd
