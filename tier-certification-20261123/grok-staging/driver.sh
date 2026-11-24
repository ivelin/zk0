#!/bin/bash
cd /home/openclaw/muse-work/zk0-tier10h-cert-20261123 || exit 1
mkdir -p grok/out
run_batch() {
  b="$1"
  [ -f "grok/out/$b.done" ] && return 0
  ~/.local/bin/grok --prompt-file "grok/prompts/$b.txt" --output-format plain \
    --disable-web-search --max-turns 80 > "grok/out/$b.raw" 2> "grok/out/$b.err"
  echo $? > "grok/out/$b.done"
}
export -f run_batch
xargs -P 3 -I{} bash -c 'run_batch "$@"' _ {} < grok/runs.txt
touch grok/GROK_SPA_RC_DONE
