import logging

from crewai import Agent

AGENT_DEFAULTS = {
    "allow_delegation": False,
    "memory": False,
    "cache": False,
    "respect_context_window": False,
}


def make_agent(role: str, goal: str, backstory: str, llm, **kwargs) -> Agent:
    logging.getLogger("orgforge.agent_factory")
    params = {**AGENT_DEFAULTS, "llm": llm}
    params.update(kwargs)
    return Agent(role=role, goal=goal, backstory=backstory, verbose=False, **params)
