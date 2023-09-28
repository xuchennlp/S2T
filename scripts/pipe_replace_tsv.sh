dir=/xuchen/st/data/must_c/en-$1

org_dir=$dir/st_tok.bak/
replace_dir=$dir/st_tok/
out_dir=$dir/st_tok/
item=audio

cp $org_dir/spm* $org_dir/config* $out_dir
sed -i "s#/mnt/bn/nas-xc-1#/xuchen/st#g" $out_dir/config*

tsv=train.tsv
./replace_tsv.sh $org_dir/$tsv $replace_dir/$tsv $out_dir/$tsv $item
tsv=dev.tsv
./replace_tsv.sh $org_dir/$tsv $replace_dir/$tsv $out_dir/$tsv $item
tsv=tst-COMMON.tsv
./replace_tsv.sh $org_dir/$tsv $replace_dir/$tsv $out_dir/$tsv $item

