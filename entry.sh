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

shell_script=$1
shift
cd `dirname ${shell_script}`
echo $@
bash ${shell_script} "$@"