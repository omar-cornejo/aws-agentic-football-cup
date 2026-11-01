# AWS Agentic Football Cup

5v5 football agents built for the Agentic Football Cup (AWS EMEA ECT Hackathon, October 2026).
Each player is its own agent: Strands Agents running on Amazon Bedrock AgentCore, one runtime per position.

The team started from the `ai-team-strands-extremely-aggressive` sample in
[aws-samples/sample-ai-possibilities](https://github.com/aws-samples/sample-ai-possibilities).
On top of it, a shared library makes most decisions in code within the engine's time limit,
and the LLM handles the rest.



<img width="1264" height="579" alt="Captura de pantalla 2026-11-01 160454" src="https://github.com/user-attachments/assets/1624a368-3daf-4a66-9801-a7645f9e0a59" />




## What's inside

```
agentic-football-sample-agents/
├── lib/                                   shared library used by every agent
├── ai-team-strands-extremely-aggressive/  hybrid team (code + LLM), GK / DEF / MID / FWD1 / FWD2
├── ai-team-llm/                           prompt-only team, the LLM decides every tick
├── sim/                                   local 5v5 simulator for evaluating strategies offline
├── deploy-strategy.sh                     deploy one strategy as its own set of runtimes
├── compare_strategies.py                  compare strategies from real match logs
└── test_*.py                              regression tests (no AWS needed)
```

## Strategies

| Strategy | Team | How it decides | Memory | Model |
|---|---|---|---|---|
| `mem2`  | hybrid | EV/Markov for the ball carrier, defensive search, GK in code, LLM for the rest | v2 | Sonnet 4.6 |
| `mem3`  | hybrid | mem2 + bench orders first, shoot zone, DEF/MID positioning | v2 | Nova 2 Lite |
| `mem31` | hybrid | mem3 + lane-gated shots, constrained carrier, lane block, loose ball | v2 | Nova 2 Lite |
| `mem32` | hybrid | mem3 + bench "shoot" rule, CENTER aim, 35 m shoot zone | v2 | Nova 2 Lite |
| `mem33` | hybrid | mem32 + DEF never past -20 m while the opponent has the ball upfield | v2 (own pool) | Nova 2 Lite |
| `alg`   | hybrid | Markov/xT + CSP defensive search + one-step lookahead | none | Nova 2 Lite |
| `llm`   | prompt-only | principles prompt + geometric facts every tick | v2 | Sonnet 4.6 |

**Memory v2** (`lib/memory_v2.py`) keeps an in-process notebook of the match in the prompt and writes events to
AgentCore Memory in the background, so lessons from earlier matches carry over to the next ones.

### Key pieces of `lib/`

- `ev_policy.py`: expected-value choice for the ball carrier (Markov / xT model fitted with `sim/fit_ev.py`).
- `def_search.py`: defensive assignment by iterative deepening search + backtracking.
- `overrides.py`: post-LLM tactical overrides (the prompts suggest, the code enforces).
- `agent_base.py`: agent factory, LLM deadline guard and fallback when the model is late.
- `units.py`: converts the live engine units to metres.
- `stamina.py`: stamina guard applied to every outgoing command.
- `mem3.py`, `mem31.py`: rules for the mem3.x strategies, switched on through `lib/tuning.*.json`.

## Results

Real matches on the cup engine:

| Strategy | Pts/match | Goals for / against per match |
|---|---|---|
| mem3  | 3.00 | 2.67 / 0.67 |
| mem32 | 2.57 | 3.14 / 2.00 |
| mem2  | 2.33 | |
| alg   | 1.40 | |
| llm   | 1.00 | |

The code-first strategies beat the prompt-only team clearly. Time per decision mattered more than model size.

## Getting started

Requirements: Python 3.10+, AWS CLI v2, [uv](https://docs.astral.sh/uv/) and the AgentCore CLI
(`pip install bedrock-agentcore-starter-toolkit strands-agents`).

```bash
cd agentic-football-sample-agents

# tests and simulator (no AWS)
python test_possession.py
python sim/strategy_eval.py --strategies mem3 mem33 --matches 5

# deploy a strategy
MEMORY_ID=<agentcore-memory-id> AWS_PROFILE=<profile> ./deploy-strategy.sh mem33
AWS_PROFILE=<profile> ./deploy-strategy.sh alg

# compare strategies from the CloudWatch logs of real matches
python compare_strategies.py --hours 6
```

## License

MIT-0. See [LICENSE](LICENSE).
