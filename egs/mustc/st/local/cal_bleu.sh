#!/usr/bin/env bash

set -e

ref=$1
gen=$2
tokenizer=$3
lang=$4
lang_pair=en-${lang}

record=$(mktemp -t temp.record.XXXXXX)
if [[ ${tokenizer} -eq 1 ]]; then
    echo "MultiBLEU" > ${record}
    cmd="multi-bleu.perl ${ref} < ${gen}"
    eval $cmd | head -n 1 >> ${record}

    cmd="detokenizer.perl -q -l ${lang} --threads 32 < ${ref} > ${ref}.detok"
    eval $cmd
    cmd="detokenizer.perl -q -l ${lang} --threads 32 < ${gen} > ${gen}.detok"
    eval $cmd
    ref=${ref}.detok
    gen=${gen}.detok
fi

echo "SacreBLEU" >> ${record}
cmd="cat ${gen} | sacrebleu ${ref} -m bleu -w 4 -l ${lang_pair}"
eval $cmd >> ${record}
cat ${record}
rm ${record}