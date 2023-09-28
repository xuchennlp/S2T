#!/usr/bin/env bash

gpu_num=1

data_tag=asr
test_subset=(dev-clean dev-other test-clean test-other all)
test_subset=(dev-clean dev-other test-clean test-other)

exp_name=
if [ "$#" -eq 1 ]; then
    exp_name=$1
fi

ctc_infer=0
n_average=10
beam_size=5
infer_ctc_weight=0.1
len_penalty=1.0
max_tokens=50000
batch_size=1
infer_debug=0
dec_model=checkpoint_best.pt

cmd="./run.sh
    --stage 2
    --stop_stage 2
    --gpu_num ${gpu_num}
    --exp_name ${exp_name}
    --n_average ${n_average}
    --beam_size ${beam_size}
    --len_penalty ${len_penalty}
    --batch_size ${batch_size}
    --max_tokens ${max_tokens}
    --dec_model ${dec_model}
    --ctc_infer ${ctc_infer}
    --infer_ctc_weight ${infer_ctc_weight}
    --infer_debug ${infer_debug}
    "

if [[ -n ${data_tag} ]]; then
    cmd="$cmd --data_tag ${data_tag}"
fi
if [[ ${#test_subset[@]} -ne 0 ]]; then
    subsets=$(echo ${test_subset[*]} | sed 's/ /,/g')
    cmd="$cmd --test_subset ${subsets}"
fi

echo $cmd
eval $cmd
