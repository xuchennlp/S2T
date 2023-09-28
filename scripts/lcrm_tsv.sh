set -e

in_tsv=$1
out_tsv=$2
item=$3

tmp=$(mktemp -t temp.record.XXXXXX)

python3 extract_txt_from_tsv.py $in_tsv $tmp $item 
cat $tmp | python3 lcrm.py > $tmp.lcrm
python3 replace_txt_from_tsv.py $in_tsv $out_tsv $tmp.lcrm $item
