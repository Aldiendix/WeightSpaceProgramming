#!/bin/bash
# GPU job queue on the box (env from tools/gpu): one job at a time, in name order.
#   queue.sh add <name> '<cmd>' [first]  |  status  |  start  |  worker (the loop itself)
Q=$WSP_WORK/queue
mkdir -p $Q/pending $Q/running $Q/done $Q/failed
case "$1" in
  add) n=$(date +%s%N | cut -c1-13); [ -n "$4" ] && n=0$n
       printf 'set -e\ncd %s\n%s\n' "$PWD" "$3" > $Q/pending/${n}_$2.sh; echo "queued ${n}_$2" ;;
  status) tail -8 $Q/queue.log; echo "running: $(ls $Q/running | grep .sh$)"; echo "pending:"; ls $Q/pending ;;
  start) pgrep -f "queue.sh worker" >/dev/null && { echo "worker already running"; exit; }
         setsid nohup bash "$0" worker > $Q/worker.out 2>&1 < /dev/null & echo started ;;
  worker) while true; do
            j=$(ls $Q/pending/*.sh 2>/dev/null | sort | head -1)
            [ -z "$j" ] && { sleep 5; continue; }
            b=$(basename $j .sh); mv $j $Q/running/
            echo "$(date +%T) START $b" >> $Q/queue.log
            if bash $Q/running/$b.sh > $Q/running/$b.log 2>&1; then r=done; else r=failed; fi
            mv $Q/running/$b.sh $Q/running/$b.log $Q/$r/; echo "$(date +%T) ${r^^} $b" >> $Q/queue.log
          done ;;
esac
