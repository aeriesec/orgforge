import logging

from crewai import Agent

AGENT_DEFAULTS = {
    "allow_delegation": False,
    "memory": False,
    "cache": False,
    "respect_context_window": False,
}


def make_agent(role: str, goal: str, backstory: str, llm, **kwargs) -> Agent:
    logger = logging.getLogger("orgforge.agent_factory")
    logger.setLevel(logging.DEBUG)
    logger.debug(
        f"\n=== AGENT CREATED ===\n"
        f"  Role:      {role}\n"
        f"  Goal:      {goal}\n"
        f"  Backstory: {backstory}\n"
        f"{'=' * 21}"
    )
    params = {**AGENT_DEFAULTS, "llm": llm}
    params.update(kwargs)
    return Agent(role=role, goal=goal, backstory=backstory, verbose=True, **params)
