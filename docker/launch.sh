#!/bin/bash

MY_IMAGE="ic-registry.epfl.ch/nlp/mete/cs503-project"

arg_job_prefix="cs503-project"
arg_job_suffix="0"
arg_job_name="$arg_job_prefix-$arg_job_suffix"

command=$1
num_gpu=${2:-1}
num_cpu=${3:-8}
gpu_memory=${4:-40G}

# Run this for train mode
if [ "$command" == "run" ]; then
	echo "Job [$arg_job_name]"

	runai submit $arg_job_name \
		-i $MY_IMAGE \
		--cpu $num_cpu \
        --gpu-memory $gpu_memory \
		--pvc runai-nlp-ismayilz-nlpdata1:/mnt/nlpdata1 \
		--pvc runai-nlp-ismayilz-scratch:/mnt/scratch \
		--command -- bash entrypoint.sh
	exit 0
fi

# Run this for interactive mode
if [ "$command" == "run_bash" ]; then
	echo "Job [$arg_job_name]"

	runai submit $arg_job_name \
		-i $MY_IMAGE \
		--cpu $num_cpu \
        --gpu-memory $gpu_memory \
		--pvc runai-nlp-ismayilz-nlpdata1:/mnt/nlpdata1 \
		--pvc runai-nlp-ismayilz-scratch:/mnt/scratch \
		--service-type=nodeport \
		--port 31122:22 \
		--interactive \
        --node-type G10 \
		--large-shm
	exit 0
fi

if [ "$command" == "log" ]; then
	runai logs $arg_job_name -f
	exit 0
fi

if [ "$command" == "stat" ]; then
	runai describe job $arg_job_name 
	exit 0
fi

if [ "$command" == "del" ]; then
	runai delete job $arg_job_name
	exit 0
fi

if [ $? -eq 0 ]; then
	runai list job
fi
