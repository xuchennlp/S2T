#!/usr/bin/env bash

# Processing AIShell ASR Datasets

# Copyright 2021 Chen Xu (xuchennlp@outlook.com)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail
export PYTHONIOENCODING=UTF-8

eval=1
time=$(date "+%m%d_%H%M")

stage=1
stop_stage=2

######## Hardware ########
# Devices
device=(0)
gpu_num=2
update_freq=1
max_tokens=100000

pwd_dir=$PWD
root_dir=${ST_ROOT}
data_root_dir=${root_dir}

code_dir=${root_dir}/S2T

# Dataset
src_lang=zh
lang=${src_lang}
dataset=aishell
data_tag=asr

task=speech_to_text
vocab_type=unigram
vocab_type=char
vocab_size=10000
speed_perturb=1
lcrm=0
tokenizer=0
use_raw_audio=0

. ./local/parse_options.sh || exit 1;

use_specific_dict=0
specific_prefix=st
specific_dir=${root_dir}/data/mustc/st
asr_vocab_prefix=spm_unigram10000_st_share

data_model_subfix=${dataset}/${data_tag}
org_data_dir=${data_root_dir}/data/${dataset}
data_dir=${data_root_dir}/data/${data_model_subfix}
train_split=train
valid_split=dev
test_split=test
test_subset=dev,test

# exp
sub_tag=
exp_prefix=$(date "+%m%d")
extra_tag=
extra_parameter=
exp_tag=baseline
exp_name=

# Training Settings
train_config=base,ctc
fp16=1
step_valid=0

# Decoding Settings
dec_model=checkpoint_best.pt
cer=1
ctc_infer=0
infer_ctc_weight=0
ctc_self_ensemble=0
ctc_inter_logit=0
n_average=10
batch_size=0
beam_size=5
len_penalty=1.0
single=0
epoch_ensemble=0
best_ensemble=1
infer_debug=0
infer_score=0
# infer_parameters="--cal-monotonic-cross-attn-weights --cal-localness --localness-window 0.1 --cal-topk-cross-attn-weights --topk-cross-attn-weights 15 --cal-entropy"

data_config=config.yaml

# Parsing Options
if [[ ${speed_perturb} -eq 1 ]]; then
    data_dir=${data_dir}_sp
    exp_prefix=${exp_prefix}_sp
fi
if [[ ${lcrm} -eq 1 ]]; then
    data_dir=${data_dir}_lcrm
    exp_prefix=${exp_prefix}_lcrm
fi
if [[ ${use_specific_dict} -eq 1 ]]; then
    data_dir=${data_dir}_${specific_prefix}
    exp_prefix=${exp_prefix}_${specific_prefix}
fi
if [[ ${tokenizer} -eq 1 ]]; then
    data_dir=${data_dir}_tok
    exp_prefix=${exp_prefix}_tok
fi
if [[ ${use_raw_audio} -eq 1 ]]; then
    data_dir=${data_dir}_raw
    exp_prefix=${exp_prefix}_raw
fi
if [[ "${vocab_type}" == "char" ]]; then
    data_dir=${data_dir}_char
    exp_prefix=${exp_prefix}_char
fi

# setup nccl envs
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE:1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0

export PATH=$PATH:${code_dir}/scripts
. ./local/parse_options.sh || exit 1;

if [[ -z ${exp_name} ]]; then
    config_string=${train_config//,/_}
    exp_name=${exp_prefix}_${config_string}_${exp_tag}
    if [[ -n ${extra_tag} ]]; then
        exp_name=${exp_name}_${extra_tag}
    fi
    if [[ -n ${exp_subfix} ]]; then
        exp_name=${exp_name}_${exp_subfix}
    fi
fi

ckpt_dir=${root_dir}/checkpoints/
model_dir=${root_dir}/checkpoints/${data_model_subfix}/${sub_tag}/${exp_name}

# Start
cd ${code_dir}
echo "Start Stage: $stage"
echo "Stop  Stage: $stop_stage"

if [[ `pip list | grep fairseq | wc -l` -eq 0 ]]; then 
    echo "Default Stage: env configure"
    pip3 install -e ${code_dir}
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "Stage -1: Data Download"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    echo "Stage 0: Data Preparation"

    if [[ ! -e ${data_dir} ]]; then
        mkdir -p ${data_dir}
    fi

    cmd="python3 ${code_dir}/examples/speech_to_text/prep_audio_data.py
        --data-root ${org_data_dir}
        --output-root ${data_dir}
        --task asr
        --src-lang ${src_lang}
        --splits ${valid_split},${test_split},${train_split}
	    --add-src
	    --share
        --vocab-type ${vocab_type}
        --vocab-size ${vocab_size}"

    if [[ ${use_raw_audio} -eq 1 ]]; then
        cmd="$cmd
        --raw"
    fi
    if [[ ${use_specific_dict} -eq 1 ]]; then
        cp -r ${specific_dir}/${asr_vocab_prefix}.* ${data_dir}
        cmd="$cmd
        --asr-prefix ${asr_vocab_prefix}"
    fi
    if [[ ${speed_perturb} -eq 1 ]]; then
        cmd="$cmd
        --speed-perturb"
    fi
    if [[ ${lcrm} -eq 1 ]]; then
        cmd="$cmd
        --lowercase-src
        --rm-punc-src"
    fi
    if [[ ${tokenizer} -eq 1 ]]; then
        cmd="$cmd
        --tokenizer"
    fi

    echo -e "\033[34mRun command: \n${cmd} \033[0m"
    [[ $eval -eq 1 ]] && eval ${cmd}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Network Training"
    [[ ! -d ${data_dir} ]] && echo "The data dir ${data_dir} is not existing!" && exit 1;

    if [[ -z ${device} || ${#device[@]} -eq 0 ]]; then
        if [[ ${gpu_num} -eq 0 ]]; then
            device=""
        else
            source ./local/utils.sh
            device=$(get_devices $gpu_num 0)
        fi
        export CUDA_VISIBLE_DEVICES=${device}
    fi

    echo -e "data=${data_dir} model=${model_dir}"

    if [[ ! -d ${model_dir} ]]; then
        mkdir -p ${model_dir}
    else
        echo "${model_dir} exists."
    fi

    cp -f ${pwd_dir}/`basename ${BASH_SOURCE[0]}` ${model_dir}
    cp -f ${pwd_dir}/train.sh ${model_dir}

    extra_parameter="${extra_parameter}
        --train-config ${pwd_dir}/conf/basis.yaml"
    cp -f ${pwd_dir}/conf/basis.yaml ${model_dir}
    config_list="${train_config//,/ }"
    idx=1
    for config in ${config_list[@]}
    do
        config_path=${pwd_dir}/conf/${config}.yaml
        if [[ ! -f ${config_path} ]]; then
            echo "No config file ${config_path}"
            exit
        fi
        cp -f ${config_path} ${model_dir}

        extra_parameter="${extra_parameter}
        --train-config${idx} ${config_path}"
        idx=$((idx + 1))
    done

    cmd="python3 -u ${code_dir}/fairseq_cli/train.py
        ${data_dir}
        --config-yaml ${data_config}
        --task ${task}
        --max-tokens ${max_tokens}
        --skip-invalid-size-inputs-valid-test
        --update-freq ${update_freq}
        --log-interval 100
        --save-dir ${model_dir}
        --tensorboard-logdir ${model_dir}"

    if [[ -n ${extra_parameter} ]]; then
        cmd="${cmd}
        ${extra_parameter}"
    fi
    if [[ ${gpu_num} -gt 0 ]]; then
        cmd="${cmd}
        --distributed-world-size $gpu_num
        --ddp-backend no_c10d"
    fi
    if [[ $fp16 -eq 1 ]]; then
        cmd="${cmd}
        --fp16"
    fi
    if [[ $step_valid -eq 1 ]]; then
        validate_interval=1
        save_interval=1
        no_epoch_checkpoints=0
        save_interval_updates=500
        keep_interval_updates=10
    fi
    if [[ -n $no_epoch_checkpoints && $no_epoch_checkpoints -eq 1 ]]; then
        cmd="$cmd
        --no-epoch-checkpoints"
    fi
    if [[ -n $validate_interval ]]; then
        cmd="${cmd}
        --validate-interval $validate_interval "
    fi
    if [[ -n $save_interval ]]; then
        cmd="${cmd}
        --save-interval $save_interval "
    fi
    if [[ -n $save_interval_updates ]]; then
        cmd="${cmd}
        --save-interval-updates $save_interval_updates"
        if [[ -n $keep_interval_updates ]]; then
        cmd="${cmd}
        --keep-interval-updates $keep_interval_updates"
        fi
    fi

    echo -e "\033[34mRun command: \n${cmd} \033[0m"

    # save info
    log=${ckpt_dir}/history.log
    echo "${time} | ${data_dir} | ${exp_name} | ${model_dir} " >> $log
    tail -n 50 ${log} > tmp.log
    mv tmp.log $log

    log=${model_dir}/train.log
    # cmd="${cmd} 2>&1 | tee -a ${log}"
    cmd="${cmd} >> ${log} 2>&1 "
    if [[ $eval -eq 1 ]]; then
        # tensorboard
        port=6666
        tensorboard --logdir ${model_dir} --port ${port} --bind_all &
    
        echo "${cmd}" > ${model_dir}/cmd
        eval $cmd
        #sleep 2s
        #tail -n "$(wc -l ${log} | awk '{print $1+1}')" -f ${log}
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Decoding"
    dec_models=
    if [[ ${n_average} -eq 1 ]]; then
        dec_models=${dec_model}
    fi
    if [[ ${n_average} -ne 1 ]]; then
        # Average models
        if [[ ${epoch_ensemble} -eq 1 ]]; then
            avg_model=avg_epoch${n_average}_checkpoint.pt

            if [[ ! -f ${model_dir}/${avg_model} ]]; then
                cmd="python3 ${code_dir}/scripts/average_checkpoints.py
                --inputs ${model_dir}
                --num-epoch-checkpoints ${n_average}
                --output ${model_dir}/${avg_model}"
                echo -e "\033[34mRun command: \n${cmd} \033[0m"
                [[ $eval -eq 1 ]] && eval $cmd
            fi
            dec_models+=(${avg_model})
        fi
        if [[ ${best_ensemble} -eq 1 ]]; then
            avg_model=avg_best${n_average}_checkpoint.pt

            if [[ ! -f ${model_dir}/${avg_model} ]]; then
                cmd="python3 ${code_dir}/scripts/average_checkpoints.py
                --inputs ${model_dir}
                --num-best-checkpoints ${n_average}
                --output ${model_dir}/${avg_model}"
                echo -e "\033[34mRun command: \n${cmd} \033[0m"
                [[ $eval -eq 1 ]] && eval $cmd
            fi
            dec_models+=(${avg_model})
        fi
    fi

    if [[ -z ${device} || ${#device[@]} -eq 0 ]]; then
        if [[ ${gpu_num} -eq 0 ]]; then
            device=""
        else
            source ./local/utils.sh
            device=$(get_devices $gpu_num 0)
        fi
        export CUDA_VISIBLE_DEVICES=${device}
    fi

    for dec_model in ${dec_models[@]}; do
        suffix=alpha${len_penalty}
        model_str=`echo $dec_model | sed -e "s#checkpoint##" | sed "s#.pt##"`
        suffix=${suffix}_${model_str}
        if [[ -n ${cer} && ${cer} -eq 1 ]]; then
            suffix=${suffix}_cer
        else
            suffix=${suffix}_wer
        fi

        suffix=${suffix}_beam${beam_size}
        if [[ ${batch_size} -ne 0 ]]; then
            suffix=${suffix}_batch${batch_size}
        else
            suffix=${suffix}_tokens${max_tokens}
        fi
        if [[ ${ctc_infer} -eq 1 ]]; then
            suffix=${suffix}_ctc
        fi
        if [[ ${ctc_self_ensemble} -eq 1 ]]; then
            suffix=${suffix}_ensemble
        fi
        if [[ ${ctc_inter_logit} -ne 0 ]]; then
            suffix=${suffix}_logit${ctc_inter_logit}
        fi
        if (( $(echo "${infer_ctc_weight} > 0" | bc -l) )); then
            suffix=${suffix}_ctc${infer_ctc_weight}
        fi
        if [[ ${infer_score} -eq 1 ]]; then
            suffix=${suffix}_score
        fi

        suffix=`echo $suffix | sed -e "s#__#_#"`
        result_file=${model_dir}/decode_result_${suffix}
        [[ -f ${result_file} ]] && rm ${result_file}

        test_subset=${test_subset//,/ }
        for subset in ${test_subset[@]}; do
            subset=${subset}
            if [[ ${infer_debug} -ne 0 ]]; then
                cmd="python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client"
            else
                cmd="python3 "
            fi
            cmd="$cmd ${code_dir}/fairseq_cli/generate.py
            ${data_dir}
            --config-yaml ${data_config}
            --gen-subset ${subset}
            --task speech_to_text
            --path ${model_dir}/${dec_model}
            --results-path ${model_dir}
            --batch-size ${batch_size}
            --max-tokens ${max_tokens}
            --beam ${beam_size}
            --lenpen ${len_penalty}
            --infer-ctc-weight ${infer_ctc_weight}
            --scoring wer"

            if [[ ${cer} -eq 1 ]]; then
                cmd="${cmd}
            --wer-char-level"
            fi
            if [[ ${ctc_infer} -eq 1 ]]; then
                cmd="${cmd}
            --ctc-infer"
            fi
            if [[ ${ctc_self_ensemble} -eq 1 ]]; then
                cmd="${cmd}
            --ctc-self-ensemble"
            fi
            if [[ ${ctc_inter_logit} -ne 0 ]]; then
                cmd="${cmd}
            --ctc-inter-logit ${ctc_inter_logit}"
            fi
            if [[ ${infer_score} -eq 1 ]]; then
                cmd="${cmd}
            --score-reference"
            fi
            if [[ -n ${infer_parameters} ]]; then
                cmd="${cmd}
            ${infer_parameters}"
            fi                        

            echo -e "\033[34mRun command: \n${cmd} \033[0m"

            cd ${code_dir}
            if [[ $eval -eq 1 ]]; then
                ctc_file=translation-${subset}.ctc
                if [[ -f ${model_dir}/${ctc_file} ]]; then
                    rm ${model_dir}/${ctc_file}
                fi

                eval $cmd
                echo "" >> ${result_file}
                tail -n 2 ${model_dir}/generate-${subset}.txt >> ${result_file}
                mv ${model_dir}/generate-${subset}.txt ${model_dir}/generate-${subset}-${suffix}.txt
                mv ${model_dir}/translation-${subset}.txt ${model_dir}/translation-${subset}-${suffix}.txt

                cd ${pwd_dir}
                if [[ -f ${model_dir}/enc_dump ]]; then
                    mv ${model_dir}/enc_dump ${model_dir}/dump-${subset}-enc-${suffix}
                fi
                if [[ -f ${model_dir}/dec_dump ]]; then
                    mv ${model_dir}/dec_dump ${model_dir}/dump-${subset}-dec-${suffix}
                fi

                trans_file=translation-${subset}-${suffix}.txt
                if [[ ${ctc_infer} -eq 1 && -f ${model_dir}/${ctc_file} ]]; then
                    ref_file=${model_dir}/${subset}.${src_lang}
                    if [[ ! -f ${ref_file} ]]; then
                        python3 ./local/extract_txt_from_tsv.py ${data_dir}/${subset}.tsv ${ref_file} "tgt_text"
                    fi
                    if [[ -f ${ref_file} ]]; then
                        ctc=$(mktemp -t temp.record.XXXXXX)
                        cd ./local
                        cmd="./cal_wer.sh ${model_dir} ${subset} ${trans_file} ${ctc_file} ${ref_file} > ${ctc}"
                        #echo $cmd
                        eval $cmd
                        cd ..

                        echo "CTC WER" >> ${result_file}
                        tail -n 2 ${ctc} >> ${result_file}

                        src_bleu=$(mktemp -t temp.record.XXXXXX)
                        cd local
                        ./cal_ctc_bleu.sh ${model_dir} ${subset} ${trans_file} ${ctc_file} ${ref_file} ${tokenizer} ${src_lang} > ${src_bleu}
                        cd ..
                        cat ${src_bleu} >> ${result_file}

                        rm ${ctc} ${src_bleu}
                    else
                        echo "No reference for source language."
                    fi
                fi
            fi
        done
        echo
        cat ${result_file}
    done
fi
