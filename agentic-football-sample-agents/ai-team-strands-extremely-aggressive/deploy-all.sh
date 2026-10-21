#!/bin/bash
# Deploy the team agents to Bedrock AgentCore.
#
#   AWS_PROFILE=your-profile ./deploy-all.sh          # all agents
#   AWS_PROFILE=your-profile ./deploy-all.sh ai-gk    # one agent
#
# Env: AGENTS_DIR, NAME_PREFIX, TUNING, MEMORY_ID, STRATEGY, OBSERVABILITY=off
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/_build${NAME_PREFIX:+-$NAME_PREFIX}"

AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_DEFAULT_REGION

ALL_AGENTS=("ai-gk" "ai-def" "ai-mid" "ai-fwd1" "ai-fwd2")

if [ -n "$1" ]; then
  AGENTS=("$1")
else
  AGENTS=("${ALL_AGENTS[@]}")
fi

echo "=========================================="
echo "  AI Team (Extremely Aggressive) — Deploy"
echo "=========================================="
echo ""

echo "Checking prerequisites..."

if ! command -v agentcore &> /dev/null; then
  echo "ERROR: 'agentcore' CLI not found."
  echo "Install: pip install bedrock-agentcore-starter-toolkit"
  exit 1
fi
echo "  agentcore CLI: OK"

if ! command -v rsync &> /dev/null; then
  echo "ERROR: 'rsync' not found."
  exit 1
fi
echo "  rsync: OK"

if ! command -v aws &> /dev/null; then
  echo "ERROR: 'aws' CLI not found."
  exit 1
fi
echo "  aws CLI: OK"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null | tr -d ' \r\n') || {
  echo "ERROR: No valid AWS credentials."
  exit 1
}
export AWS_ACCOUNT_ID
echo "  AWS Account: $AWS_ACCOUNT_ID"
echo "  AWS Region:  $AWS_DEFAULT_REGION"
echo ""

cleanup() {
  echo ""
  echo "Cleaning up build directory..."
  rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

DEPLOYED=()
FAILED=()

AGENTS_DIR="${AGENTS_DIR:-$SCRIPT_DIR}"

for agent in "${AGENTS[@]}"; do
  AGENT_SRC="$AGENTS_DIR/$agent"
  STAGE="$BUILD_DIR/$agent"

  echo "=========================================="
  echo "  Deploying: $agent"
  echo "=========================================="

  if [ ! -d "$AGENT_SRC" ]; then
    echo "  ERROR: Agent directory not found: $AGENT_SRC"
    FAILED+=("$agent")
    continue
  fi

  rm -rf "$STAGE"
  mkdir -p "$STAGE/src"

  cp "$AGENT_SRC/src/main.py" "$STAGE/src/main.py"
  rsync -a --exclude='__pycache__' "$SCRIPT_DIR/../lib/" "$STAGE/lib/"
  if [ -n "${TUNING:-}" ]; then
    cp "$SCRIPT_DIR/../lib/$TUNING" "$STAGE/lib/tuning.json"
    echo "  tuning: lib/$TUNING"
  fi
  cp "$AGENT_SRC/requirements.txt" "$STAGE/requirements.txt"

  sed \
    -e "s|\${AWS_ACCOUNT_ID}|$AWS_ACCOUNT_ID|g" \
    -e "s|\${AWS_DEFAULT_REGION}|$AWS_DEFAULT_REGION|g" \
    "$AGENT_SRC/.bedrock_agentcore.yaml.template" > "$STAGE/.bedrock_agentcore.yaml"
  if [ -n "${NAME_PREFIX:-}" ]; then
    sed -i "s/agg_/${NAME_PREFIX}/g" "$STAGE/.bedrock_agentcore.yaml"
  fi
  RUNTIME_NAME=$(sed -n 's/^default_agent:[[:space:]]*//p' "$STAGE/.bedrock_agentcore.yaml")
  ENV_ARGS=()
  if [ -n "${MEMORY_ID:-}" ]; then
    ENV_ARGS=(--env "MEMORY_ID=$MEMORY_ID" --env "TEAM_ID=${TEAM_ID:-neptune}"
              --env "AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION")
  fi
  if [ -n "${STRATEGY:-}" ]; then
    ENV_ARGS+=(--env "STRATEGY=$STRATEGY")
  fi
  if [ "${OBSERVABILITY:-on}" = "off" ]; then
    sed -i 's/^\([[:space:]]*enabled:[[:space:]]*\)true/\1false/' "$STAGE/.bedrock_agentcore.yaml"
  fi

  echo "  Deploying from: $STAGE"
  if (cd "$STAGE" && agentcore deploy --auto-update-on-conflict "${ENV_ARGS[@]}"); then
    echo "  ✅ $agent: DEPLOYED ($RUNTIME_NAME)"
    DEPLOYED+=("$agent")
    if [ -n "${MEMORY_ID:-}" ]; then
      RID=$(aws bedrock-agentcore-control list-agent-runtimes \
        --query "agentRuntimes[?agentRuntimeName=='$RUNTIME_NAME'].agentRuntimeId" --output text)
      ROLE=$(aws bedrock-agentcore-control get-agent-runtime --agent-runtime-id "$RID" \
        --query roleArn --output text | sed 's#.*/##')
      aws iam put-role-policy --role-name "$ROLE" --policy-name AgentCoreMemoryAccess \
        --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"bedrock-agentcore:ListEvents\",\"bedrock-agentcore:CreateEvent\",\"bedrock-agentcore:GetEvent\",\"bedrock-agentcore:DeleteEvent\",\"bedrock-agentcore:RetrieveMemoryRecords\",\"bedrock-agentcore:GetMemoryRecord\",\"bedrock-agentcore:ListMemoryRecords\",\"bedrock-agentcore:CreateSession\",\"bedrock-agentcore:GetSession\",\"bedrock-agentcore:ListSessions\"],\"Resource\":\"arn:aws:bedrock-agentcore:${AWS_DEFAULT_REGION}:${AWS_ACCOUNT_ID}:memory/*\"}]}" \
        && echo "  ✅ memory permissions -> $ROLE" || echo "  ⚠️  could not attach memory permissions to $ROLE"
    fi
  else
    echo "  ❌ $agent: FAILED"
    FAILED+=("$agent")
  fi
  echo ""
done

echo "=========================================="
echo "  Deployment Summary"
echo "=========================================="
echo ""
echo "  Deployed: ${DEPLOYED[*]:-none}"
echo "  Failed:   ${FAILED[*]:-none}"
echo "  Account:  $AWS_ACCOUNT_ID"
echo "  Region:   $AWS_DEFAULT_REGION"
echo ""

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Some agents failed to deploy. Check the output above."
  exit 1
fi

echo "All agents deployed successfully."
