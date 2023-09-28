#!/usr/bin/env bash

# calculate wmt14 en-de multi-bleu score

if [ $# -ne 1 ]; then
    echo "usage: $0 GENERATE_PY_OUTPUT"
    exit 1
fi
echo -e "\n RUN >> "$0

requirement_scripts=(detokenizer.perl replace-unicode-punctuation.perl tokenizer.perl multi-bleu.perl)
for script in ${requirement_scripts[@]}; do
    if ! which ${script} > /dev/null; then
        echo "Error: it seems that moses is not installed or exported int the environment variables." >&2
        return 1
    fi
done

detokenizer=detokenizer.perl
replace_unicode_punctuation=replace-unicode-punctuation.perl
tokenizer=tokenizer.perl
multi_bleu=multi-bleu.perl

GEN=$1
SYS=$GEN.sys
REF=$GEN.ref

cat $GEN | cut -f 3 > $REF
cat $GEN | cut -f 4 > $SYS

#detokenize the decodes file to format the manner to do tokenize
$detokenizer -l de < $SYS > $SYS.dtk
$detokenizer -l de < $REF > $REF.dtk

#replace unicode
$replace_unicode_punctuation -l de < $SYS.dtk > $SYS.dtk.punc
$replace_unicode_punctuation -l de < $REF.dtk > $REF.dtk.punc

#tokenize the decodes file by moses tokenizer.perl
$tokenizer -l de < $SYS.dtk.punc > $SYS.dtk.punc.tok
$tokenizer -l de < $REF.dtk.punc > $REF.dtk.punc.tok

#"rich-text format" --> rich ##AT##-##AT## text format.
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $SYS.dtk.punc.tok > $SYS.dtk.punc.tok.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $REF.dtk.punc.tok > $REF.dtk.punc.tok.atat

$multi_bleu $REF.dtk.punc.tok.atat < $SYS.dtk.punc.tok.atat

rm -f $SYS.dtk $SYS.dtk.punc $SYS.dtk.punc.tok $REF.dtk $REF.dtk.punc $REF.dtk.punc.tok