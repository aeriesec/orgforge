# token_tracker.py
from crewai.events import BaseEventListener, CrewKickoffCompletedEvent, crewai_event_bus

class OrgForgeTokenListener(BaseEventListener):
    """
    Listens to every LLM call CrewAI makes and logs token usage to MongoDB.
    Active only when Memory is initialised with debug_tokens=True.
    Single instance covers PLANNER_MODEL and WORKER_MODEL transparently.
    """

    def __init__(self):
        super().__init__()
        self._mem = None   # injected after Memory is constructed

    def attach(self, mem) -> None:
        """Call this once in OrgForgeFlow.__init__ after Memory is ready."""
        self._mem = mem

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def on_llm_completed(source, event: CrewKickoffCompletedEvent):
            if self._mem is None:
                return
            # 1. Extract the agents from the Crew (the source)
            agents = getattr(source, "agents", [])

            # 2. Grab the GOAL from the first agent to use as our tracker
            primary_goal = "unknown_goal"
            if agents:
                primary_goal = getattr(agents[0], "goal", "unknown_goal")

            # 3. Extract the model
            model_name = "unknown"
            if agents and getattr(agents[0], "llm", None):
                llm = agents[0].llm
                model_name = getattr(llm, "model", getattr(llm, "model_name", "unknown"))

            prompt_tokens = getattr(event, "prompt_tokens", 0)
            completion_tokens = getattr(event, "completion_tokens", 0)

            # Fallback: if they are 0, check inside the Crew's usage_metrics or event.metrics
            if prompt_tokens == 0 and completion_tokens == 0:
                usage = getattr(source, "usage_metrics", getattr(event, "metrics", {}))
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                else:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0)
                    completion_tokens = getattr(usage, "completion_tokens", 0)

            current_day = getattr(self._mem, "_current_day", 0) or 1
            self._mem.log_token_usage(
                    caller            = primary_goal,
                    call_type         = "llm",
                    model             = model_name,
                    day               = current_day,
                    timestamp         = event.timestamp.isoformat(),
                    prompt_tokens     = prompt_tokens,
                    completion_tokens = completion_tokens,
                    total_tokens      = event.total_tokens,
                    extra             = {
                                            "agent_role": getattr(agents[0], "role", "") if agents else "",
                                        }
                )

# Module-level instance — must exist for handlers to be registered
# (CrewAI garbage-collects listeners with no live reference)
orgforge_token_listener = OrgForgeTokenListener()