#!/usr/bin/env bash
set -e 

splits=(train dev tst-COMMON)
langs=(en zh)
dir=$1
cd $dir

export PATH=$PATH:/root/st/Fairseq-S2T/scripts

for split in ${splits[@]}; do
    for lang in ${langs[@]}; do
        in=$split/txt/${split}.${lang}
        out=$split/txt/${split}.tok.$lang
        cmd="token.sh $lang $in $out"
        echo $cmd
        eval $cmd
    done
done
