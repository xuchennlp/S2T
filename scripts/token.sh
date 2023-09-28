#!/usr/bin/env bash
set -e 

lang=$1
in_file=$2
out_file=$3

export PATH=$PATH:$PWD

if [[ $lang == "ja" ]] ; then
    cmd="mecab_token.py $in_file $out_file"
elif [[ $lang == "zh" ]] ; then
	cmd="python3 -m jieba -d ' ' $in_file > $out_file"
else
    cmd="tokenizer.perl -l ${lang} --threads 32 -no-escape < ${in_file} > ${out_file}"
fi
echo $cmd
eval $cmd
