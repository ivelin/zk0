#!/usr/bin/env bash
# Pull rc2 sheets + episodes csvs from the DGX in 10-sheet chunks.
set -u
cd "$(dirname "$0")"
DGX=~/workspace/tools/dgx.sh
pull() { # prefix, file list...
  local prefix=$1; shift
  timeout 25 $DGX "cd /home/openclaw/muse-work/zk0-tier10h-cert-20261123/sheets && tar -cf - $* | base64 -w0" | base64 -d > .pull-chunk.tar 2>/dev/null
  tar -xf .pull-chunk.tar -C sheets/ 2>/dev/null && echo "OK $prefix $*" >> pull-rc2.log || echo "FAIL $prefix $*" >> pull-rc2.log
}
rm -f pull-rc2.log
for spec in "rc-stack-bowls 100" "rc-stack-cubes 100" "bb-push-green 39" "bb-flip-left 20"; do
  set -- $spec; p=$1; n=$2
  for start in $(seq 0 10 $((n-1))); do
    files=""
    for i in $(seq $start $((start+9))); do [ $i -lt $n ] && files="$files ${p}__ep${i}.jpg"; done
    pull $p $files
  done
  timeout 25 $DGX "cat /home/openclaw/muse-work/zk0-tier10h-cert-20261123/sheets/${p}_episodes.csv" > ${p}_episodes.csv 2>/dev/null && echo "CSV OK $p" >> pull-rc2.log
done
rm -f .pull-chunk.tar
echo PULL_DONE >> pull-rc2.log
