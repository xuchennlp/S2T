#!/usr/bin/env bash

THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd ${THIS_DIR}
export ST_ROOT=/xuchen/st
export NCCL_DEBUG=INFO

echo "nameserver 114.114.114.114" >> /etc/resolv.conf

if [[ `pip list | grep fairseq | wc -l` -eq 0 ]]; then 
    echo "default stage: env configure"
    pip3 install -e .
fi

all_cmd=$1
all_cmd_dir=`dirname ${all_cmd}`
echo "cmd dir: $all_cmd_dir"
echo_flag=1
pre_num=-1

cp $all_cmd ${all_cmd}.bak
while :
do 
    line=`head -n1 $all_cmd`
    if [[ -z $line ]]; then
        record=$(mktemp -t temp.record.XXXXXX)
        gpustat > $record
        all_devices=$(seq 0 "$(sed '1,2d' ${record} | wc -l)");

        device=()
        count=0
        for dev in ${all_devices[@]}
        do
            item=$((dev + 2))
            use=$(head -n $item ${record} | tail -1 | cut -d '|' -f3 | cut -d '/' -f1)

            if [[ $use -lt 100 ]]; then
                device[$count]=$dev
                count=$((count + 1))
                if [[ $count -eq $gpu_num ]]; then
                    break
                fi
            fi
        done

        if [[ $echo_flag -eq 1 ]]; then
            echo "No cmd. Current free GPU: ${count}. Waiting."
            echo_flag=0
        fi
        sleep 300s
        continue
    fi
    gpu_num=$(echo $line | awk '{print $1}')
    shell_script=$(echo $line | awk '{print $2}')
    cmd=$(echo $line | awk '{$1=""; print $0}')

    echo_flag=1
    while :
    do
        record=$(mktemp -t temp.record.XXXXXX)
        gpustat > $record
        all_devices=$(seq 0 "$(sed '1,2d' ${record} | wc -l)");

        device=()
        count=0
        for dev in ${all_devices[@]}
        do
            item=$((dev + 2))
            use=$(head -n $item ${record} | tail -1 | cut -d '|' -f3 | cut -d '/' -f1)

            if [[ $use -lt 100 ]]; then
                device[$count]=$dev
                count=$((count + 1))
                if [[ $count -eq $gpu_num ]]; then
                    break
                fi
            fi
        done
        if [[ ${#device[@]} -lt $gpu_num ]]; then
            if [[ ${pre_num} -ne ${gpu_num} ]]; then
            	echo "Current free GPU: ${count}, need GPU: ${gpu_num}. Waiting."
	        fi
            pre_num=$gpu_num
            sleep 300s
        else
            echo "Run $cmd"
            cd `dirname ${shell_script}`
            avail_devices=$(echo $(IFS=','; echo "${device[*]}"))
            
            time=$(date "+%m%d_%H%M")
            echo "Time: $time | Devices: $avail_devices | $cmd" >> ${all_cmd}.record
            echo $line >> ${all_cmd}.run
            sed -i '1d' ${all_cmd}

            export CUDA_VISIBLE_DEVICES=$avail_devices
            eval $cmd &
            cd ${THIS_DIR}
            sleep 300s
        fi
        break
    done
done
wait
echo "all done"
echo "all done" >> ${all_cmd}.record
