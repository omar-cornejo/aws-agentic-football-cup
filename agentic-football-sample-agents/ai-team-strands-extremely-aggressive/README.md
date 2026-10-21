# Hybrid team (code + LLM)

Five agents, one per position, each deployed as its own AgentCore runtime:

| Agent | Player | Role |
|---|---|---|
| `ai-gk`   | P0 | goalkeeper |
| `ai-def`  | P1 | defender |
| `ai-mid`  | P2 | central midfielder |
| `ai-fwd1` | P3 | left striker (1-1-2) |
| `ai-fwd2` | P4 | right striker |
| `ai-mid2` | P3 | left midfielder, only for the 1-2-1 formation |

Every tick the agent runs the code-first layer from `../lib` (EV carrier, defensive search, overrides,
stamina guard) and only asks the LLM when the code has no decision. If the model does not answer within
the deadline, the fallback command is sent.

The strategy (`mem2`, `mem3`, `mem31`, `mem32`, `mem33`, `alg`) is chosen at deploy time by the tuning file:

```bash
MEMORY_ID=<memory-id> ../deploy-strategy.sh mem33
./deploy-all.sh ai-def      # redeploy one agent with the default lib/tuning.json
```
