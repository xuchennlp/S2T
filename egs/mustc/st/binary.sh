set -e

eval=1

lcrm=1
tokenizer=0

vocab_type=unigram
vocab_size=5000
use_raw_audio=0
speed_perturb=0

dataset=mustc
root_dir=~/st
code_dir=${root_dir}/Fairseq-S2T
org_data_dir=${root_dir}/data/${dataset}
data_dir=${root_dir}/data/${dataset}/st

use_specific_dict=0
specific_prefix=st
specific_dir=${root_dir}/data/mustc/st
asr_vocab_prefix=spm_unigram10000_st_share

src_lang=en
tgt_lang=zh
subsets=(2019)

splits=$(echo ${subsets[*]} | sed 's/ /_/g')
cmd="python ${code_dir}/examples/speech_to_text/prep_audio_data.py
    --data-root ${org_data_dir}
    --output-root ${data_dir}
    --task asr
    --src-lang ${src_lang}
    --tgt-lang ${tgt_lang}
    --splits ${splits}
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
