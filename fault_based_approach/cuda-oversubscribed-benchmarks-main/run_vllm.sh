#!/bin/bash
source .venv/bin/activate

VLLM_PATH="/data/pranjal/cuda-benchmarks/vllm/.venv/bin/"
n_prompts=5000

# There are two tokenizers: tokenizers.Tokenizer and vllm/tokenizers.
# Change the working directory to avoid a clash.
# Solution credits: ChatGPT (diagnosis) and myself (the cd solution).
cd -

if [ ! -d "$VLLM_PATH" ]; then
    echo "Where is the venv? Check $VLLM_PATH"
fi

kill_vllm_server () {
	for pid in `pgrep -i ^vllm`
	do
        echo "Killing PID $pid"
        kill -9 $pid
	done

    for t in `seq 1 60`; do
        if [ -z "$(lsof -i :8000)" ]; then
            echo "Shutdown: $t seconds"
            return
        fi
        sleep 1
    done
    echo "ERROR: VLLM server did not shut down in 60 s"
    nvidia-smi
    lsof -i :8000
}

check_port_available () {
	if [ -n "$(lsof -i :8000)" ]; then
		echo "ERROR: Port occupied"
		lsof -i :8000
		exit 5
	fi
}

wait_for_startup () {
	for t in `seq 1 60`; do
		if [ -z "$(lsof -i :8000)" ]; then
			sleep 1
		else
			echo "Server startup: $t seconds"
			return
		fi
	done

	echo "Server did not start: exiting"
	exit 3
}


if [ ! -d  vllm ]; then
	echo "Expecting a vllm directory under `pwd`."
	# exit 2
fi

if [[ "$1" == "kill" ]]; then
	echo "Killing server"
	kill_vllm_server
	exit 0
fi

check_port_available

# cd vllm
echo In PWD "$PWD"
export VLLM_USE_UVM=2
export TIME="%C: %e"

echo "Using this VLLM installation: '`which vllm`'"

/bin/time vllm serve facebook/opt-125m &
wait_for_startup

/bin/time vllm bench serve --model facebook/opt-125m --num-prompts $n_prompts --max-concurrency 15

kill_vllm_server

echo "** COMPLETED BENCHMARKS **"
