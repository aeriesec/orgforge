import logging
from typing import Dict, Union
from config_loader import COMPANY_NAME, COMPANY_DESCRIPTION, PERSONAS, DEFAULT_PERSONA
from utils.helpers import dept_of_name

logger = logging.getLogger("orgforge.persona_utils")


def get_voice_card(
    names: Union[str, list], context: str = "general", graph_dynamics=None, mem=None
) -> str:
    """
    Unified persona generator for all OrgForge LLM prompts.
    Combines identity, tenure, expertise, style, and dynamic mood.
    Accepts a single name or a list of names. Identical personas are deduplicated.
    """
    is_single = isinstance(names, str)
    name_list = [names] if is_single else names

    card_to_names = {}
    name_to_history = {}

    for name in name_list:
        p = PERSONAS.get(name, DEFAULT_PERSONA)
        stress = graph_dynamics._stress.get(name, 30) if graph_dynamics else 30
        quirks = p.get("typing_quirks", "standard professional grammar")
        tenure = p.get("tenure", "mid")
        expertise = (
            ", ".join(str(e) for e in p.get("expertise", [])[:3])
            or "general engineering"
        )
        social_role = p.get("social_role", "Contributor")
        dept = dept_of_name(name)
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

        if context == "async":
            lines.insert(2, f"  Expertise: {expertise}")
        elif context == "watercooler":
            lines.insert(2, f"  Personal interests: {interests}")

        _anti_pattern_contexts = {"async", "design", "collision", "dm"}
        if anti_patterns and context in _anti_pattern_contexts:
            lines.append(f"  Never write {placeholder} as: {anti_patterns.strip()}")

        _pet_peeve_contexts = {"design", "collision"}
        if pet_peeves and context in _pet_peeve_contexts:
            lines.append(f"  Pet peeves (will react if triggered): {pet_peeves}")

        identity_block = "\n".join(lines)

        sections = [
            f"IDENTITY: You are {placeholder} ({tenure} tenure). Role: {p.get('social_role', 'Contributor')}.",
            f"COMPANY: You work at {COMPANY_NAME}, which {COMPANY_DESCRIPTION}.",
            f"{identity_block}\n\nNever acknowledge being an AI. Stay in character.",
        ]

        template = "\n".join(sections)

        # Group identical templates
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
