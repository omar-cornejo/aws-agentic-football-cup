#!/bin/bash
# Deploy one strategy as its own set of 5 runtimes (1-1-2).
#
#   MEMORY_ID=<memory-id> ./deploy-strategy.sh mem33
#   ./deploy-strategy.sh alg               # no memory needed
#   ./deploy-strategy.sh llm ai-mid        # one agent only
#
# Strategies: mem2 mem3 mem31 mem32 mem33 alg llm
# Runtimes:   <strategy>_gk_agent, _def_agent, _mid_agent, _fwd1_agent, _fwd2_agent
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
MEM="${MEMORY_ID:-}"
case "$1" in
  mem2) export AGENTS_DIR="$ROOT/ai-team-strands-extremely-aggressive" NAME_PREFIX=mem2_ \
               TUNING=tuning.mem2.json MEMORY_ID="$MEM" STRATEGY=mem2 ;;
  mem3) export AGENTS_DIR="$ROOT/ai-team-strands-extremely-aggressive" NAME_PREFIX=mem3_ \
               TUNING=tuning.mem3.json MEMORY_ID="$MEM" STRATEGY=mem3 ;;
  mem31) export AGENTS_DIR="$ROOT/ai-team-strands-extremely-aggressive" NAME_PREFIX=mem31_ \
               TUNING=tuning.mem31.json MEMORY_ID="$MEM" STRATEGY=mem31 ;;
  mem32) export AGENTS_DIR="$ROOT/ai-team-strands-extremely-aggressive" NAME_PREFIX=mem32_ \
               TUNING=tuning.mem32.json MEMORY_ID="$MEM" STRATEGY=mem32 ;;
  mem33) export AGENTS_DIR="$ROOT/ai-team-strands-extremely-aggressive" NAME_PREFIX=mem33_ \
               TUNING=tuning.mem33.json MEMORY_ID="$MEM" STRATEGY=mem33 ;;
  alg)  export AGENTS_DIR="$ROOT/ai-team-strands-extremely-aggressive" NAME_PREFIX=alg_ \
               TUNING=tuning.alg.json STRATEGY=alg; unset MEMORY_ID ;;
  llm)  export AGENTS_DIR="$ROOT/ai-team-llm" NAME_PREFIX=llm_ \
               TUNING=tuning.llm.json MEMORY_ID="$MEM" STRATEGY=llm ;;
  *) echo "usage: $0 mem2|mem3|mem31|mem32|mem33|alg|llm [ai-gk|ai-def|ai-mid|ai-fwd1|ai-fwd2]"; exit 1 ;;
esac
if [ "$STRATEGY" != "alg" ] && [ -z "$MEM" ]; then
  echo "MEMORY_ID is required for strategy $1"; exit 1
fi
echo "Strategy $1 -> runtimes ${NAME_PREFIX}*  (tuning lib/$TUNING, memory ${MEMORY_ID:-off})"
exec "$ROOT/ai-team-strands-extremely-aggressive/deploy-all.sh" "${@:2}"
