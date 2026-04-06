import logging
from typing import Dict, Union
from config_loader import COMPANY_NAME, COMPANY_DESCRIPTION, PERSONAS, DEFAULT_PERSONA
from utils.helpers import dept_of_name

logger = logging.getLogger("orgforge.persona_utils")

DEPARTMENT_EXPERTISE_DEFAULTS = {
    "Engineering_Backend": ["general backend", "code review", "documentation"],
    "Engineering_Mobile": ["general mobile", "testing", "tickets"],
    "Product": ["requirements", "stakeholder updates", "roadmap admin"],
    "Sales_Marketing": ["CRM hygiene", "outreach", "reporting"],
    "HR_Ops": ["scheduling", "documentation", "vendor comms"],
    "Design": ["asset delivery", "feedback cycles", "figma"],
    "QA_Support": ["test cases", "bug triage", "customer comms"],
}


class PersonaUtils:
    def __init__(self):
        self._graph_dynamics = None
        self._mem = None
        self._crm = None

    def configure(self, graph_dynamics=None, mem=None, crm=None):
        if graph_dynamics is not None:
            self._graph_dynamics = graph_dynamics
        if mem is not None:
            self._mem = mem
        if crm is not None:
            self._crm = crm

    def get_voice_card(
        self,
        names: Union[str, list],
        context: str = "general",
        graph_dynamics=None,
        mem=None,
        internal=True,
    ) -> str:
        """
        Unified persona generator for all OrgForge LLM prompts.
        Combines identity, tenure, expertise, style, and dynamic mood.
        Accepts a single name or a list of names. Identical personas are deduplicated.
        """

        if not internal:
            return self._get_external_voice_card(names, context, mem)

        is_single = isinstance(names, str)
        name_list = [names] if is_single else names

        card_to_names = {}
        name_to_history = {}

        for name in name_list:
            p = PERSONAS.get(name) or DEFAULT_PERSONA
            stress = graph_dynamics._stress.get(name, 30) if graph_dynamics else 30
            quirks = p.get("typing_quirks", "standard professional grammar")
            tenure = p.get("tenure", "mid")
            dept = dept_of_name(name)

            expertise = ", ".join(
                str(e) for e in p.get("expertise", [])[:3]
            ) or ", ".join(
                DEPARTMENT_EXPERTISE_DEFAULTS.get(
                    dept, ["cross-functional communication"]
                )
            )
            social_role = p.get("social_role", "Contributor")

            interests = (
                ", ".join(
                    str(i) for i in (p.get("interests") or p.get("expertise", []))[:3]
                )
                or "general topics"
            )
            style = p.get("style", "")
            anti_patterns = p.get("anti_patterns", "")
            pet_peeves = p.get("pet_peeves", "")

            if mem:
                past = mem.persona_history(name, n=2)
                if past:
                    name_to_history[name] = " | ".join(
                        f"Day {e.day}: {e.summary}" for e in past
                    )

            _moods: Dict[str, tuple] = {
                "one_on_one": (
                    "drained, short replies",
                    "a bit distracted",
                    "relaxed and present",
                ),
                "async": (
                    "visibly stressed, terse replies, wants to resolve this fast",
                    "somewhat distracted but trying to help",
                    "engaged and happy to dig in",
                ),
                "design": (
                    "terse, wants to decide fast and move on",
                    "engaged but watching the clock",
                    "thinking carefully, happy to explore trade-offs",
                ),
                "mentoring": (
                    "drained, keeping answers short",
                    "patient but distracted",
                    "engaged and generous with their time",
                ),
                "collision": (
                    "visibly stressed, terse, wants this resolved immediately",
                    "frustrated but trying to stay professional",
                    "measured and collegial",
                ),
                "dm": (
                    "stressed and frustrated, wants this unblocked now",
                    "concerned but calm",
                    "helpful and focused",
                ),
                "watercooler": (
                    "visibly drained, short replies, clearly wants this to be over quickly",
                    "a bit distracted, somewhat engaged but mind is elsewhere",
                    "relaxed and happy to take a break",
                ),
                "general": (
                    "drained, short replies",
                    "a bit distracted",
                    "relaxed and present",
                ),
            }
            high, mid, low = _moods.get(context, _moods["general"])
            mood = high if stress > 80 else mid if stress > 60 else low

            placeholder = "__NAMES_PLACEHOLDER__"

            if context == "one_on_one":
                header = f"{placeholder} | Tenure: {tenure}"
            elif context == "async":
                header = f"{placeholder} | Tenure: {tenure} | Dept: {dept}"
            elif context == "design":
                header = f"{placeholder} | Role: {social_role} | Expertise: {expertise}"
            elif context == "mentoring":
                header = f"{placeholder} | Tenure: {tenure} | Expertise: {expertise}"
            elif context == "collision":
                header = f"{placeholder} | Dept: {dept} | Role: {social_role}"
            elif context == "dm":
                header = f"{placeholder}"
            elif context == "watercooler":
                header = f"{placeholder} | Tenure: {tenure} | Role: {social_role}"
            else:
                header = f"{placeholder} | Tenure: {tenure}"

            if style and context != "watercooler":
                header += f" | Style: {style}"

            lines = [header, f"  Typing style: {quirks}", f"  Current mood: {mood}"]

            lines.insert(2, f"  Expertise: {expertise}")

            if context == "watercooler":
                lines.insert(2, f"  Personal interests: {interests}")

            _anti_pattern_contexts = {"async", "design", "collision", "dm"}
            if anti_patterns and context in _anti_pattern_contexts:
                lines.append(f"  Never write {placeholder} as: {anti_patterns.strip()}")

            _pet_peeve_contexts = {"design", "collision"}
            if pet_peeves and context in _pet_peeve_contexts:
                lines.append(f"  Pet peeves (will react if triggered): {pet_peeves}")

            _crm_hint = self.crm_pressure_hint(name, self._crm)
            if _crm_hint:
                lines.append(_crm_hint)

            identity_block = "\n".join(lines)

            sections = [
                f"IDENTITY: You are {placeholder} ({tenure} tenure). Role: {p.get('social_role', 'Contributor')}.",
                f"COMPANY: You work at {COMPANY_NAME}, which {COMPANY_DESCRIPTION}.",
                f"{identity_block}\n\nNever acknowledge being an AI. Stay in character.",
            ]

            template = "\n".join(sections)

            if template not in card_to_names:
                card_to_names[template] = []
            card_to_names[template].append(name)

        parts = []
        for template, group_names in card_to_names.items():
            combined_names = " / ".join(group_names)

            final_card = template.replace("__NAMES_PLACEHOLDER__", combined_names)

            history_lines = []
            for n in group_names:
                if n in name_to_history:
                    prefix = f"{n}: " if len(group_names) > 1 else ""
                    history_lines.append(f"{prefix}{name_to_history[n]}")

            if history_lines:
                final_card += "\n\nRECENT HISTORY:\n" + "\n".join(history_lines)

            if is_single:
                return final_card
            else:
                parts.append(f"PERSONA(S) FOR {combined_names}:\n{final_card}")

        return "\n\n".join(parts)

    def crm_pressure_hint(self, name: str, crm) -> str:
        """
        Returns a short LLM directive reflecting this person's live CRM exposure.
        Injected into get_voice_card() when crm is passed. Empty string if no signal.

        Examples:
        "Jordan owns an at-risk $85K deal in Negotiation — anxious about the close."
        "Sam is the liaison for a customer with 2 Urgent support tickets open."
        """
        from crm_system import NullCRMSystem

        if crm is None or isinstance(crm, NullCRMSystem):
            return ""

        hints = []

        owned_opps = list(
            crm._sf_o.find(
                {
                    "owner": name,
                    "stage": {"$nin": ["Closed Won", "Closed Lost"]},
                },
                {
                    "account_name": 1,
                    "stage": 1,
                    "amount": 1,
                    "risk_notes": 1,
                    "probability": 1,
                },
            )
        )
        for opp in owned_opps[:2]:
            stage = opp.get("stage", "")
            amt = opp.get("amount", 0)
            org = opp.get("account_name", "a customer")
            if opp.get("risk_notes"):
                hints.append(
                    f"{name} owns an at-risk ${amt:,} deal with {org} ({stage}) — "
                    f"privately anxious about losing it."
                )
            elif stage == "Negotiation/Review":
                hints.append(
                    f"{name} is deep in contract negotiation with {org} (${amt:,}) — "
                    f"excited but checking email constantly."
                )

        if not hints:
            urgent_count = crm._zd.count_documents(
                {"status": {"$in": ["Open", "Pending"]}, "priority": "Urgent"}
            )
            normal_count = crm._zd.count_documents(
                {"status": {"$in": ["Open", "Pending"]}, "priority": {"$ne": "Urgent"}}
            )
            if urgent_count > 0:
                hints.append(
                    f"{name} is aware there {'is' if urgent_count == 1 else 'are'} "
                    f"{urgent_count} Urgent support ticket(s) open — "
                    f"feels pressure to resolve the customer situation."
                )
            elif normal_count >= 3:
                hints.append(
                    f"{name} has {normal_count} open support tickets in the queue — "
                    f"background stress from customer load."
                )

        return "  CRM pressure: " + " | ".join(hints) if hints else ""

    def _get_external_voice_card(
        self,
        names: Union[str, list],
        context: str,
        mem,
    ) -> str:
        """
        Builds a voice card for an external contact (vendor or customer).
        Pulls persona data from the inbound_email_sources collection rather
        than the internal PERSONAS config.
        """
        is_single = isinstance(names, str)
        name_list = [names] if is_single else names

        m = mem or self._mem
        if m is None:
            return f"You are {name_list[0]}. Respond professionally."

        if not hasattr(self, "_external_sources_cache"):
            self._external_sources_cache = None
        if self._external_sources_cache is None:
            self._external_sources_cache = m.get_inbound_email_sources() or []

        source_map = {s["name"]: s for s in self._external_sources_cache}

        parts = []
        for name in name_list:
            source = source_map.get(name)
            if not source:
                parts.append(
                    f"You are {name}, an external contact. Respond professionally."
                )
                continue

            persona = source.get("persona", {})
            quirks = persona.get("typing_quirks", source.get("tone", "professional"))
            role = source.get("contact_role", persona.get("social_role", "Contact"))
            archetype = source.get("persona_archetype", "")
            expertise = ", ".join(persona.get("expertise", []))
            sentiment = source.get("sentiment_baseline", 0.7)
            history = source.get("history_summary", "")
            org = source.get("org", name)
            category = source.get("category", "contact")

            if sentiment < 0.4:
                mood = "skeptical and guarded, looking for reasons to escalate"
            elif sentiment < 0.6:
                mood = "neutral but watchful, needs reassurance"
            else:
                mood = "collaborative and constructive"

            card = "\n".join(
                filter(
                    None,
                    [
                        f"IDENTITY: You are {name}, {role} at {org}.",
                        f"ARCHETYPE: {archetype}" if archetype else "",
                        f"EXPERTISE: {expertise}" if expertise else "",
                        f"Typing style: {quirks}",
                        f"Current mood: {mood}",
                        f"Relationship history: {history}" if history else "",
                        f"Never acknowledge being an AI or a simulation. "
                        f"You are a real {category} contact.",
                    ],
                )
            )
            parts.append(card)

        return "\n\n".join(parts)


persona_utils = PersonaUtils()
