#!/usr/bin/env bash

# Processing MuST-C Datasets

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
gpu_num=8
update_freq=1

pwd_dir=$PWD
root_dir=${ST_ROOT}
data_root_dir=${root_dir}

code_dir=${root_dir}/S2T

# Dataset
src_lang=en
tgt_lang=de
dataset=must_c
data_tag=st

task=speech_to_text
vocab_type=unigram
asr_vocab_size=5000
vocab_size=10000
share_dict=1
speed_perturb=0
lcrm=0
tokenizer=0
use_raw_audio=0

. ./local/parse_options.sh || exit 1;
lang=${src_lang}-${tgt_lang}

use_specific_dict=0
specific_prefix=valid
specific_dir=${data_root_dir}/data/${dataset}/${lang}/st
asr_vocab_prefix=spm_unigram10000_st_share
st_vocab_prefix=spm_unigram10000_st_share

data_model_subfix=${dataset}/${lang}/${data_tag}
org_data_dir=${data_root_dir}/data/${dataset}/${lang}
data_dir=${data_root_dir}/data/${data_model_subfix}

train_split=train
valid_split=dev
test_split=tst-COMMON
test_subset=dev,tst-COMMON

# Exp
sub_tag=
exp_prefix=$(date "+%m%d")
extra_tag=
extra_parameter=
exp_tag=baseline
exp_name=

# Training Settings
train_config=base,ctc
fp16=1
max_tokens=40000
step_valid=0
bleu_valid=0

# Decoding Settings
batch_size=0
sacrebleu=1
dec_model=checkpoint_best.pt
ctc_infer=0
infer_ctc_weight=0
n_average=10
beam_size=5
len_penalty=1.0
infer_debug=0
infer_score=0
#infer_parameters="--cal-monotonic-cross-attn-weights --cal-localness --localness-window 0.1 --cal-topk-cross-attn-weights --topk-cross-attn-weights 15 --cal-entropy"

# Parsing Options
if [[ ${share_dict} -eq 1 ]]; then
	data_config=config_share.yaml
else
	data_config=config.yaml
fi
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
    echo "Stage 0: ASR Data Preparation"
    if [[ ! -e ${data_dir} ]]; then
        mkdir -p ${data_dir}
    fi

    # create ASR vocabulary if necessary
    cmd="python3 ${code_dir}/examples/speech_to_text/prep_audio_data.py
        --data-root ${org_data_dir}
        --output-root ${data_dir}/asr4st
        --task asr
        --raw
        --src-lang ${src_lang}
        --splits ${valid_split},${test_split},${train_split}
        --vocab-type ${vocab_type}
        --vocab-size ${asr_vocab_size}"
    if [[ ${lcrm} -eq 1 ]]; then
        cmd="$cmd
        --lowercase-src
        --rm-punc-src"
    fi
    if [[ ${tokenizer} -eq 1 ]]; then
        cmd="$cmd
        --tokenizer"
    fi
    if [[ $eval -eq 1 && ${share_dict} -ne 1 && ${use_specific_dict} -ne 1 ]]; then
        echo -e "\033[34mRun command: \n${cmd} \033[0m"
        mkdir -p ${data_dir}/asr4st
        eval $cmd
        asr_prefix=spm_${vocab_type}${asr_vocab_size}_asr
        cp -f ${data_dir}/asr4st/${asr_prefix}* ${data_dir}
    fi

    echo "Stage 0: ST Data Preparation"
    cmd="python3 ${code_dir}/examples/speech_to_text/prep_audio_data.py
        --data-root ${org_data_dir}
        --output-root ${data_dir}
        --task st
        --add-src
        --src-lang ${src_lang}
        --tgt-lang ${tgt_lang}
        --splits ${valid_split},${test_split},${train_split}
        --cmvn-type utterance
        --vocab-type ${vocab_type}
        --vocab-size ${vocab_size}"

    if [[ ${use_raw_audio} -eq 1 ]]; then
        cmd="$cmd
        --raw"
    fi
    if [[ ${use_specific_dict} -eq 1 ]]; then
        cp -r ${specific_dir}/${asr_vocab_prefix}.* ${data_dir}
        cp -r ${specific_dir}/${st_vocab_prefix}.* ${data_dir}
        if [[ $share_dict -eq 1 ]]; then
            cmd="$cmd
        --share
        --st-spm-prefix ${st_vocab_prefix}"
        else
            cmd="$cmd
        --st-spm-prefix ${st_vocab_prefix}
        --asr-prefix ${asr_vocab_prefix}"
        fi
    else
        if [[ $share_dict -eq 1 ]]; then
            cmd="$cmd
        --share"
        else
            cmd="$cmd
        --asr-prefix ${asr_prefix}"
        fi
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
        --source-lang ${src_lang}
        --target-lang ${tgt_lang}
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
    if [[ $bleu_valid -eq 1 ]]; then
        cmd="$cmd
        --eval-bleu
        --eval-bleu-args '{\"beam\": 1}'
        --eval-tokenized-bleu
        --eval-bleu-remove-bpe
        --best-checkpoint-metric bleu
        --maximize-best-checkpoint-metric"
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
    if [[ ${n_average} -ne 1 ]]; then
        # Average models
		dec_model=avg_${n_average}_checkpoint.pt

        if [[ ! -f ${model_dir}/${dec_model} ]]; then
            cmd="python3 ${code_dir}/scripts/average_checkpoints.py
            --inputs ${model_dir}
            --num-best-checkpoints ${n_average}
            --output ${model_dir}/${dec_model}"
            echo -e "\033[34mRun command: \n${cmd} \033[0m"
            [[ $eval -eq 1 ]] && eval $cmd
        fi
	else
		dec_model=${dec_model}
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

    suffix=alpha${len_penalty}
    model_str=`echo $dec_model | sed -e "s#checkpoint##" | sed "s#.pt##"`
    suffix=${suffix}_${model_str}
    if [[ ${sacrebleu} -eq 1 ]]; then
        suffix=${suffix}_sacrebleu
    else
        suffix=${suffix}_multibleu
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
        --skip-invalid-size-inputs-valid-test
        --infer-ctc-weight ${infer_ctc_weight}
        --lenpen ${len_penalty}"

        if [[ ${ctc_infer} -eq 1 ]]; then
            cmd="${cmd}
        --ctc-infer"
        fi
        if [[ ${sacrebleu} -eq 1 ]]; then
            cmd="${cmd}
        --scoring sacrebleu"
            if [[ "${tgt_lang}" = "ja" ]]; then
                cmd="${cmd}
        --sacrebleu-tokenizer ja-mecab"
            elif [[ "${tgt_lang}" == "zh" ]]; then
                cmd="${cmd}
        --sacrebleu-tokenizer zh"
            fi
            if [[ ${tokenizer} -eq 1 ]]; then
                cmd="${cmd}
        --tokenizer moses
        --source-lang ${src_lang}
        --target-lang ${tgt_lang}"
            fi
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
            if [[ ${ctc_infer} -eq 1 && -f ${model_dir}/${ctc_file} ]]; then
                rm ${model_dir}/${ctc_file}
            fi
            xctc_file=translation-${subset}.xctc
            if [[ ${ctc_infer} -eq 1 && -f ${model_dir}/${xctc_file} ]]; then
                rm ${model_dir}/${xctc_file}
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
                    python3 ./local/extract_txt_from_tsv.py ${data_dir}/${subset}.tsv ${ref_file} "src_text"
                fi
                if [[ -f ${ref_file} ]]; then
                    ctc=$(mktemp -t temp.record.XXXXXX)
                    cd ./local
                    ./cal_wer.sh ${model_dir} ${subset} ${trans_file} ${ctc_file} ${ref_file} > ${ctc}
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

            xctc_file=translation-${subset}.xctc
            if [[ ${ctc_infer} -eq 1 && -f ${model_dir}/${xctc_file} ]]; then
                ref_file=${model_dir}/${subset}.${tgt_lang}
                if [[ ! -f ${ref_file} ]]; then
                    python3 ./local/extract_txt_from_tsv.py ${data_dir}/${subset}.tsv ${ref_file} "tgt_text"
                fi
                if [[ -f ${ref_file} ]]; then
                    xctc=$(mktemp -t temp.record.XXXXXX)
                    cd local
                    ./cal_wer.sh ${model_dir} ${subset} ${trans_file} ${xctc_file} ${ref_file} > ${xctc}
                    cd ..

                    echo "XCTC WER" >> ${result_file}
                    tail -n 2 ${xctc} >> ${result_file}

                    tgt_bleu=$(mktemp -t temp.record.XXXXXX)
                    cd local
                    ./cal_ctc_bleu.sh ${model_dir} ${subset} ${trans_file} ${xctc_file} ${ref_file} ${tokenizer} ${tgt_lang} > ${tgt_bleu}
                    cd ..
                    cat ${tgt_bleu} >> ${result_file}

                    rm ${xctc} ${tgt_bleu}
                else
                    echo "No reference for target language."
                fi
            fi
        fi
	done
	echo
    cat ${result_file}
fi
