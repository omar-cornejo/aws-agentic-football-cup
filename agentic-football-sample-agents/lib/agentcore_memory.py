"""AgentCore Memory (short-term) for the LLM ticks, one memory session per match."""
from __future__ import annotations

import os
import re

import tuning as tuning_mod


def enabled() -> bool:
    return bool(tuning_mod.get("agentcore_memory", False)) and bool(os.environ.get("MEMORY_ID"))


def _safe(s: str) -> str:
    """Memory session / actor ids: letters, digits, '-', '_' only, <= 100 chars."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", s)[:100]


def build(base_agent, session_key: str, position_label: str):
    """A copy of base_agent (same model, prompt, limits) backed by Memory."""
    from strands import Agent
    from strands.agent.conversation_manager import SlidingWindowConversationManager
    from strands.models import BedrockModel
    from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
    from bedrock_agentcore.memory.integrations.strands.session_manager import (
        AgentCoreMemorySessionManager)

    cfg = dict(base_agent.model.config)
    model = BedrockModel(model_id=cfg.get("model_id"),
                         temperature=cfg.get("temperature", 0.2),
                         max_tokens=cfg.get("max_tokens", 200))
    window = int(tuning_mod.get("memory_window_ticks", 5))
    team = os.environ.get("TEAM_ID", "neptune")
    sm = AgentCoreMemorySessionManager(
        agentcore_memory_config=AgentCoreMemoryConfig(
            memory_id=os.environ["MEMORY_ID"],
            session_id=_safe(f"{session_key}-{position_label}"),
            actor_id=_safe(f"{team}-{position_label}"),
            batch_size=2,
        ),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    )
    return Agent(model=model, system_prompt=base_agent.system_prompt,
                 session_manager=sm,
                 conversation_manager=SlidingWindowConversationManager(window_size=window * 2),
                 callback_handler=None)
