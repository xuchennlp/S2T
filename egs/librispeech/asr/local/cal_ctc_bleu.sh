#!/usr/bin/env bash

set -e

infer_dir=$1
tag=$2
s2s_infer_file=${infer_dir}/$3
org_ctc_infer_file=${infer_dir}/$4
ref=$5
tokenizer=$6
lang=$7

idx=${infer_dir}/${tag}_idx
ctc_infer=${infer_dir}/${tag}_ctc_infer
ctc_infer_sort=${infer_dir}/${tag}_ctc_infer_sort

if [[ ! -f ${ctc_infer_sort} ]]; then
    cut -f1 ${s2s_infer_file} > ${idx}
    paste ${idx} ${org_ctc_infer_file} > ${ctc_infer}
    sort -n -t $'\t' ${ctc_infer} | cut -f2 > ${ctc_infer_sort}
fi

gen=${ctc_infer_sort}
./cal_bleu.sh ${ref} ${gen} ${tokenizer} ${lang}