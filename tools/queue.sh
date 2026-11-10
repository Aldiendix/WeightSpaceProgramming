#!/bin/bash
# GPU job queue on the box (env from tools/gpu): one worker per GPU, each runs one job at a time, in name order.
#   queue.sh add <name> '<cmd>' [first]  |  status  |  start (one worker per GPU)  |  worker <gpu> (the loop)
Q=$WSP_WORK/queue
mkdir -p $Q/pending $Q/running $Q/done $Q/failed
case "$1" in
  add) n=$(date +%s%N | cut -c1-13); [ -n "$4" ] && n=0$n
       printf 'set -e\ncd %s\n%s\n' "$PWD" "$3" > $Q/pending/${n}_$2.sh; echo "queued ${n}_$2" ;;
  status) tail -8 $Q/queue.log; echo "running:"; ls $Q/running | grep .sh$; echo "pending:"; ls $Q/pending ;;
  start) for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
           pgrep -f "queue.sh worker $g\$" >/dev/null && { echo "gpu $g: worker already running"; continue; }
           setsid nohup bash "$0" worker $g > $Q/worker_$g.out 2>&1 < /dev/null & echo "gpu $g: started"
         done ;;
  worker) export CUDA_VISIBLE_DEVICES=$2
          while true; do
            j=$(ls $Q/pending/*.sh 2>/dev/null | sort | head -1)
            [ -z "$j" ] && { sleep 5; continue; }
            b=$(basename $j .sh)
            mv $j $Q/running/ 2>/dev/null || continue  # another worker took it first
            echo "$(date +%T) START $b (gpu $2)" >> $Q/queue.log
            if bash $Q/running/$b.sh > $Q/running/$b.log 2>&1; then r=done; else r=failed; fi
            mv $Q/running/$b.sh $Q/running/$b.log $Q/$r/; echo "$(date +%T) ${r^^} $b" >> $Q/queue.log
          done ;;
esac
