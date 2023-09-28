gpu_num=4
cmd="sh train.sh"

while :
do
    record=$(mktemp -t temp.record.XXXXXX)
    gpustat > $record
    all_devices=$(seq 0 "$(sed '1,2d' ${record} | wc -l)");

    count=0
    for dev in ${all_devices[@]}
    do
        line=$((dev + 2))
        use=$(head -n $line ${record} | tail -1 | cut -d '|' -f3 | cut -d '/' -f1)

        if [[ $use -lt 100 ]]; then
            device[$count]=$dev
            count=$((count + 1))
            if [[ $count -eq $gpu_num ]]; then
                break
            fi
        fi
    done
    if [[ ${#device[@]} -lt $gpu_num ]]; then
        sleep 60s
    else
        echo "Run $cmd"
        eval $cmd
        sleep 10s
        exit
    fi
done
