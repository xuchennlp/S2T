dir=

cmd=""
for d in `ls $dir`; do
    echo $d
    ./run.sh --stage 3 --max_tokens 50000 --infer_parameter "--cal_flops" --exp_name $d
done