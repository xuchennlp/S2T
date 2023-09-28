set -e

org_tsv=$1
replace_tsv=$2
out_tsv=$3
item=$4

tmp=$(mktemp -t temp.record.XXXXXX)

python3 extract_txt_from_tsv.py $replace_tsv $tmp $item 
python3 replace_txt_from_tsv.py $org_tsv $out_tsv $tmp $item
