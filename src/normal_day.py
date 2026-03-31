from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from agent_factory import make_agent
from config_loader import COMPANY_DESCRIPTION
from crewai import Process, Task, Crew

from crm_system import NullCRMSystem
from json_repair import json_repair
from memory import Memory, SimEvent
from graph_dynamics import GraphDynamics
from planner_models import (
    AgendaItem,
    DepartmentDayPlan,
    EngineerDayPlan,
    OrgDayPlan,
    ProposedEvent,
)
from causal_chain_handler import CausalChainHandler
from insider_threat import _NullInjector
from utils.persona_utils import persona_utils

logger = logging.getLogger("orgforge.normalday")


class NormalDayHandler:
    def __init__(
        self,
        config,
        mem: Memory,
        state,
        graph_dynamics: GraphDynamics,
        social_graph,
        git,
        worker_llm,
        planner_llm,
        clock,
        persona_helper,
        confluence_writer=None,
        vader=None,
        threat_injector=None,
        embed_worker=None,
        lifecycle=None,
        crm=None,
    ):
        self._config = config
        self._mem = mem
        self._state = state
        self._gd = graph_dynamics
        self._graph = social_graph
        self._git = git
        self._worker = worker_llm
        self._planner = planner_llm
        self._base = config["simulation"].get("output_dir", "./export")
        self._domain = config["simulation"]["domain"]
        self._company = config["simulation"]["company_name"]
        self._all_names = [n for dept in config["org_chart"].values() for n in dept]
        self._org_chart = config["org_chart"]
        self._clock = clock
        self._persona_helper = persona_helper
        self._confluence = confluence_writer
        self._registry = getattr(confluence_writer, "_registry", None)
        self._vader = vader
        self._threat = threat_injector or _NullInjector()
        self._embed_worker = embed_worker
        self._lifecycle = lifecycle
        self._crm = crm or NullCRMSystem()

    def handle(self, org_plan: OrgDayPlan) -> None:
        """Processes both planned agenda items and unplanned org collisions."""
        logger.info("  [bold blue]💬 Normal Day Activity[/bold blue]")
        date_str = str(self._state.current_date.date())

        self._execute_agenda_items(org_plan, date_str)

        for event in org_plan.collision_events:
            self._handle_collision_event(event, date_str)

        self._fire_sales_outreach(date_str)

        self._maybe_bot_alerts()
        self._maybe_adhoc_confluence()

    def _execute_agenda_items(self, org_plan: OrgDayPlan, date_str: str) -> None:
        """
        Walk every engineer's agenda across all departments sequentially.
        """
        all_participants: List[str] = []
        seen_discussions: set = set()

        ordered_depts = sorted(
            org_plan.dept_plans.keys(),
            key=lambda d: 0 if "engineering" in d.lower() else 1,
        )

        for dept in ordered_depts:
            dept_plan = org_plan.dept_plans[dept]

            for eng_plan in dept_plan.engineer_plans:
                watercooler_prob = self._config["simulation"].get(
                    "watercooler_prob", 0.15
                )
                will_be_distracted = random.random() < watercooler_prob
                distraction_fired = False

                non_deferred_indices = [
                    idx for idx, item in enumerate(eng_plan.agenda) if not item.deferred
                ]
                distraction_index = (
                    random.choice(non_deferred_indices)
                    if non_deferred_indices
                    else None
                )

                for idx, item in enumerate(eng_plan.agenda):
                    if item.deferred:
                        self._log_deferred_item(eng_plan.name, item, date_str)
                        continue

                    if (
                        will_be_distracted
                        and not distraction_fired
                        and idx == distraction_index
                    ):
                        self._trigger_watercooler_chat(eng_plan.name, date_str)
                        penalty_hours = random.uniform(0.16, 0.25)
                        item.estimated_hrs += penalty_hours
                        self._clock.advance_actor(eng_plan.name, penalty_hours)
                        distraction_fired = True

                    if item.activity_type in (
                        "design_discussion",
                        "mentoring",
                        "1on1",
                        "async_question",
                        "pr_review",
                    ):
                        collaborators = (
                            list(item.collaborator) if item.collaborator else []
                        )
                        participant_set = frozenset([eng_plan.name] + collaborators)

                        if item.activity_type == "design_discussion":
                            key = (
                                "design_discussion",
                                participant_set,
                                item.description,
                            )
                        else:
                            key = (item.activity_type, participant_set)

                        if key in seen_discussions:
                            continue
                        seen_discussions.add(key)

                    try:
                        participants = self._dispatch(
                            eng_plan, item, dept_plan, date_str
                        )
                        all_participants.extend(participants)
                    except Exception as exc:
                        logger.error(
                            f"[normal_day] {eng_plan.name}/{item.activity_type} failed: {exc}",
                            exc_info=True,
                        )

        if all_participants:
            unique = list(set(all_participants))
            self.graph_dynamics_record(unique)

    def _dispatch(
        self,
        eng_plan: EngineerDayPlan,
        item: AgendaItem,
        dept_plan: DepartmentDayPlan,
        date_str: str,
    ) -> List[str]:
        """Route an agenda item to the right handler. Returns actors involved."""

        t = item.activity_type

        if t == "ticket_progress":
            return self._handle_ticket_progress(eng_plan, item, date_str)
        elif t == "pr_review":
            return self._handle_pr_review(eng_plan, item, date_str)
        elif t == "1on1":
            return self._handle_one_on_one(eng_plan, item, date_str)
        elif t == "async_question":
            return self._handle_async_question(eng_plan, item, dept_plan, date_str)
        elif t == "design_discussion":
            return self._handle_design_discussion(eng_plan, item, dept_plan, date_str)
        elif t == "mentoring":
            return self._handle_mentoring(eng_plan, item, date_str)
        elif t == "deep_work":
            self._log_deep_work(eng_plan.name, item, date_str)
            return [eng_plan.name]
        elif t == "code_review_comment":
            return self._handle_pr_review(eng_plan, item, date_str)
        else:
            return self._handle_generic_activity(eng_plan, item, date_str)

    def _handle_ticket_progress(
        self,
        eng_plan: EngineerDayPlan,
        item: AgendaItem,
        date_str: str,
    ) -> List[str]:
        """
        Simulates a team member working on a JIRA ticket.

        Engineering tickets → comment + optional PR spawn (existing behaviour).
        Non-engineering tickets → comment + completion artifact (Confluence page,
        email, or Slack thread). Determined by ticket["dept_type"] stamped at
        sprint planning time. Causal chain is tracked for both paths so evals
        work across all departments.
        """
        assignee = eng_plan.name
        ticket_id = item.related_id
        if not ticket_id:
            return []

        current_actor_time, new_cursor = self._clock.advance_actor(assignee, hours=2.0)
        current_actor_time_iso = current_actor_time.isoformat()

        ticket = self._mem.get_ticket(ticket_id, as_of_time=current_actor_time_iso)
        if not ticket:
            return [eng_plan.name]

        if self._try_force_merge_stale_pr(ticket, assignee, date_str):
            return [assignee]

        dept_type = ticket.get("dept_type", "eng")
        is_non_eng = dept_type == "non_eng"
        completion_artifact = ticket.get("completion_artifact", "slack")

        ctx = self._mem.context_for_person(
            name=assignee,
            as_of_time=current_actor_time_iso,
            n=2,
        )

        if self._registry:
            ticket_ctx = self._registry.ticket_summary(
                ticket, self._state.day
            ).for_prompt()
        else:
            ticket_ctx = (
                f"Ticket: [{ticket_id}] {ticket.get('title', '')}\n"
                f"Status: {ticket.get('status', 'To Do')}\n"
                f"Recent comments: "
                + (
                    "\n".join(
                        f"  - {c['author']} ({c['date']}): {c['text']}"
                        for c in ticket.get("comments", [])[-3:]
                    )
                    or "None."
                )
            )

        backstory = persona_utils.get_voice_card(assignee, "async", self._gd, self._mem)

        # Surface any orphaned domains this ticket touches so the engineer's
        # comment naturally reflects uncertainty about legacy systems
        orphaned_domain_hint = ""
        if self._lifecycle:
            all_domains = list(
                self._mem._db["domain_registry"].find({"primary_owner": None})
            )
            ticket_text = (
                f"{ticket.get('title', '')} {ticket.get('description', '')}".lower()
            )
            for rec in all_domains:
                if any(tag in ticket_text for tag in rec.get("system_tags", [])):
                    pct = int(rec.get("documentation_coverage", 0) * 100)
                    known_by = rec.get("known_by", [])
                    orphaned_domain_hint += (
                        f"\n⚠ NOTE: '{rec['domain']}' is underdocumented ({pct}% coverage). "
                        f"Former owner: {rec.get('former_owner', 'unknown')}. "
                        f"If your work touches this system, reflect genuine uncertainty in your comment."
                        + (
                            f" Others with partial knowledge: {', '.join(known_by)}."
                            if known_by
                            else ""
                        )
                    )

        if is_non_eng:
            persona = self._config.get("personas", {}).get(assignee, {})
            agent_role = persona.get("role", dept_of_name(assignee, self._org_chart))
            task_complete_field = "is_task_complete"
            task_complete_hint = (
                "true only if the full action item is done and ready for review"
            )
            completion_note = (
                f"This is a non-engineering action item. "
                f"Your completion artifact will be a {completion_artifact}. "
                f"Do NOT mention code or PRs."
            )
        else:
            agent_role = "Software Engineer"
            task_complete_field = "is_code_complete"
            task_complete_hint = (
                "true only if the full coding phase is finished "
                "(false on day 1 of a complex ticket)"
            )
            completion_note = ""

        reviewer_feedback_hint = ""
        if ticket.get("status") == "In Review":
            for linked_pr_id in ticket.get("linked_prs", []):
                linked_pr = self._mem._prs.find_one(
                    {
                        "pr_id": linked_pr_id,
                        "status": "open",
                        "created_at": {"$lte": current_actor_time_iso},
                    },
                    {"_id": 0},
                )
                if linked_pr and linked_pr.get("changes_requested"):
                    recent_feedback = linked_pr.get("comments", [])[-3:]
                    if recent_feedback:
                        feedback_lines = "\n".join(
                            f"  - {c['author']}: {c['text'][:200]}"
                            for c in recent_feedback
                        )
                        reviewer_feedback_hint = (
                            f"\nREVIEWER FEEDBACK TO ADDRESS:\n{feedback_lines}\n"
                            f"Your comment must describe specifically how you addressed this feedback. "
                            f"Do not describe unrelated work.\n"
                        )
                    break

        agent = make_agent(
            role=f"{assignee} — {agent_role}",
            backstory=backstory,
            goal="Make progress on the ticket and report status.",
            llm=self._worker,
        )
        task = Task(
            description=(
                f"You are {assignee}. You worked on ticket [{ticket_id}] today.\n\n"
                f"Your task today: {item.description}\n"
                f"{reviewer_feedback_hint}"
                + (orphaned_domain_hint + "\n" if orphaned_domain_hint else "")
                + f"IMPORTANT: Your comment must be specifically about this ticket's work — "
                f"do not describe unrelated tasks.\n"
                f"{completion_note}\n\n"
                f"Respond ONLY with valid JSON. No preamble, no markdown fences.\n"
                f"{{\n"
                f'  "comment": "string — 1-3 sentences describing what you did today, '
                f'written as a JIRA comment in your own voice",\n'
                f'  "{task_complete_field}": boolean — {task_complete_hint}\n'
                f"}}\n\n"
                f"--- TICKET CONTEXT ---\n"
                f"{ticket_ctx}\n\n"
                f"--- MEMORY CONTEXT ---\n"
                f"{ctx}"
            ),
            expected_output="Valid JSON only. No preamble, no markdown fences.",
            agent=agent,
        )

        raw_result = str(
            Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        ).strip()
        clean_json = raw_result.replace("```json", "").replace("```", "").strip()

        try:
            parsed_data = json.loads(clean_json)
            if isinstance(parsed_data, list):
                parsed_data = parsed_data[0] if parsed_data else {}
            elif not isinstance(parsed_data, dict):
                parsed_data = {}
            comment_text = parsed_data.get("comment", f"Worked on {ticket_id}.")
            is_complete = parsed_data.get(
                task_complete_field,
                parsed_data.get("is_code_complete", False),
            )
        except json.JSONDecodeError:
            comment_text = clean_json
            is_complete = False

        BLOCKER_KEYWORDS = (
            "blocked",
            "blocker",
            "waiting on",
            "can't proceed",
            "stuck",
        )
        if any(kw in comment_text.lower() for kw in BLOCKER_KEYWORDS):
            self._mem.log_event(
                SimEvent(
                    type="blocker_flagged",
                    timestamp=current_actor_time_iso,
                    day=self._state.day,
                    date=date_str,
                    actors=[assignee],
                    artifact_ids={"jira": ticket_id},
                    facts={"ticket_id": ticket_id, "comment": comment_text},
                    summary=f"{assignee} flagged a blocker on {ticket_id}.",
                    tags=["jira", "blocker"],
                )
            )
        ticket.setdefault("comments", []).append(
            {
                "author": assignee,
                "date": date_str,
                "created": current_actor_time_iso,
                "updated": current_actor_time_iso,
                "text": f'"{comment_text}"',
                "day": self._state.day,
            }
        )

        if ticket.get("status") == "In Review":
            for linked_pr_id in ticket.get("linked_prs", []):
                linked_pr = self._mem._prs.find_one(
                    {
                        "pr_id": linked_pr_id,
                        "status": "open",
                        "created_at": {"$lte": current_actor_time_iso},
                    },
                    {"_id": 0},
                )
                if linked_pr and linked_pr.get("changes_requested"):
                    linked_pr["changes_requested"] = False
                    self._mem.upsert_pr(linked_pr)

        if ticket["status"] == "To Do":
            ticket["status"] = "In Progress"
            if "in_progress_since" not in ticket:
                ticket["in_progress_since"] = self._state.day

        from causal_chain_handler import CausalChainHandler

        active_inc = next(
            (i for i in self._state.active_incidents if i.ticket_id == ticket_id),
            None,
        )

        existing_chain = ticket.get("causal_chain", [ticket_id])
        chain = CausalChainHandler(existing_chain[0])
        for artifact in existing_chain[1:]:
            chain.append(artifact)

        if active_inc and getattr(active_inc, "causal_chain", None):
            chain = active_inc.causal_chain
        else:
            existing_chain = ticket.get("causal_chain", [ticket_id])
            chain = CausalChainHandler(existing_chain[0])
            for artifact in existing_chain[1:]:
                chain.append(artifact)

        comment_id = f"{ticket_id}_comment_{len(ticket['comments'])}"
        chain.append(comment_id)

        spawned_pr_id = None
        completion_id = None

        ticket_age = self._state.day - ticket.get("in_progress_since", self._state.day)
        force_complete = ticket_age >= 3
        actor_incident_bound = any(
            eng_plan.is_on_call and i.ticket_id != ticket_id
            for i in self._state.active_incidents
        )

        if is_complete or force_complete:
            if is_non_eng:
                completion_id = self._complete_non_eng_ticket(
                    ticket,
                    assignee,
                    comment_text,
                    ctx,
                    date_str,
                    current_actor_time.isoformat(),
                    chain,
                )
            else:
                spawned_pr_id = self._complete_eng_ticket(
                    ticket,
                    assignee,
                    actor_incident_bound,
                    date_str,
                    current_actor_time.isoformat(),
                    chain,
                    is_complete,
                )

        ticket["causal_chain"] = chain.snapshot()
        ticket["updated_at"] = current_actor_time_iso

        for pr_id in ticket.get("linked_prs", []):
            pr = self._mem._prs.find_one(
                {
                    "pr_id": pr_id,
                    "status": "open",
                    "created_at": {"$lte": current_actor_time_iso},
                },
                {"_id": 0},
            )
            if pr and pr.get("changes_requested"):
                pr["changes_requested"] = False
                ticket["status"] = "In Review"
                ticket["in_review_since"] = self._state.day
                self._mem.upsert_pr(pr)

        self._save_ticket(ticket)

        ticket_body = "\n".join(
            filter(
                None,
                [
                    ticket.get("title", ""),
                    ticket.get("description", ""),
                    ticket.get("root_cause", ""),
                    "\n".join(c.get("text", "") for c in ticket.get("comments", [])),
                ],
            )
        )

        self._mem.embed_artifact(
            id=ticket_id,
            type="jira",
            title=ticket.get("title", ticket_id),
            content=ticket_body,
            day=self._state.day,
            date=date_str,
            timestamp=current_actor_time_iso,
            metadata={
                "assignee": ticket.get("assignee", ""),
                "status": ticket["status"],
                "dept_type": dept_type,
            },
        )

        self._mem.embed_artifact(
            id=comment_id,
            type="jira_comment",
            title=f"Comment on {ticket_id}",
            content=comment_text,
            day=self._state.day,
            date=date_str,
            timestamp=current_actor_time_iso,
            metadata={
                "ticket_id": ticket_id,
                "author": assignee,
                "dept_type": dept_type,
            },
        )

        artifacts = {"jira": ticket_id, "jira_comment": comment_id}
        if spawned_pr_id:
            artifacts["pr"] = spawned_pr_id
        if completion_id:
            artifacts[completion_artifact] = completion_id

        facts = {
            "ticket_id": ticket_id,
            "status": ticket["status"],
            "dept_type": dept_type,
            "causal_chain": chain.snapshot(),
        }
        if spawned_pr_id:
            facts["spawned_pr"] = spawned_pr_id
        if completion_id:
            facts["completion_artifact"] = completion_id

        summary = f"{assignee} worked on {ticket_id}."
        if spawned_pr_id:
            summary += f" Opened PR {spawned_pr_id}."
        elif completion_id:
            summary += f" Completed → {completion_id}."

        self._mem.log_event(
            SimEvent(
                type="ticket_progress",
                timestamp=current_actor_time_iso,
                day=self._state.day,
                date=date_str,
                actors=[assignee],
                artifact_ids=artifacts,
                facts=facts,
                summary=summary,
                tags=["jira", "engineering"],
            )
        )

        bucket = self._state.ticket_actors_today.setdefault(ticket_id, set())
        bucket.add(assignee)

        if self._vader:
            self._score_and_apply_sentiment(comment_text, [assignee], self._vader)

        generated_artifacts = [ticket_id]
        if spawned_pr_id:
            generated_artifacts.append(spawned_pr_id)
        if completion_id:
            generated_artifacts.append(completion_id)

        return generated_artifacts

    def _try_force_merge_stale_pr(
        self, ticket: dict, assignee: str, date_str: str
    ) -> bool:
        """
        Evaluates if a ticket currently 'In Review' should be force-merged.
        Returns True if the ticket was processed (merged or left idling), False otherwise.
        """
        if ticket.get("status") != "In Review":
            return False

        if ticket.get("status") == "In Review":
            linked_prs = ticket.get("linked_prs", [])
            review_age = self._state.day - ticket.get(
                "in_review_since", ticket.get("in_progress_since", self._state.day)
            )
            actor_clock_ok = self._clock.now(assignee).hour < 17

            open_pr_with_changes = bool(linked_prs) and any(
                self._mem._prs.find_one(
                    {"pr_id": p, "status": "open", "changes_requested": True},
                    {"_id": 0, "pr_id": 1},
                )
                for p in linked_prs
            )

            force_merge = (
                review_age >= 5
                and bool(linked_prs)
                and actor_clock_ok
                and not open_pr_with_changes
            )

            if force_merge:
                current_actor_time, _ = self._clock.advance_actor(assignee, hours=0.2)
                current_actor_time_iso = current_actor_time.isoformat()

                stale_pr = next(
                    (
                        doc
                        for p in linked_prs
                        for doc in [
                            self._mem._prs.find_one(
                                {"pr_id": p, "status": "open"}, {"_id": 0}
                            )
                        ]
                        if doc
                    ),
                    None,
                )

                if stale_pr:
                    self._handle_pr_review_for_incident(
                        reviewer=assignee,
                        pr=stale_pr,
                        date_str=date_str,
                        timestamp=current_actor_time_iso,
                    )

                    stale_pr = (
                        self._mem._prs.find_one(
                            {"pr_id": stale_pr["pr_id"]}, {"_id": 0}
                        )
                        or stale_pr
                    )
                    stale_pr["status"] = "merged"

                    import os
                    import json as _json

                    pr_path = f"{self._base}/git/prs/{stale_pr['pr_id']}.json"
                    os.makedirs(os.path.dirname(pr_path), exist_ok=True)
                    with open(pr_path, "w") as f:
                        _json.dump(stale_pr, f, indent=2)
                    self._mem.upsert_pr(stale_pr)

                    self._git.merge_pr(stale_pr["pr_id"])

                    ticket["status"] = "Done"
                    ticket["completed_at"] = current_actor_time_iso
                    self._save_ticket(ticket)

                    self._emit_bot_message(
                        "engineering",
                        "GitHub Actions",
                        f"✅ Auto-merged {stale_pr['pr_id']} after {review_age} days in review: "
                        f"{stale_pr.get('title', '')[:80]}",
                        current_actor_time_iso,
                    )
                    logger.info(
                        f"    [green]✅ Force-merged {stale_pr['pr_id']} — "
                        f"{ticket.get('id')} → Done (age={review_age}d)[/green]"
                    )

                return True

        return False

    def _complete_non_eng_ticket(
        self,
        ticket: dict,
        assignee: str,
        comment_text: str,
        ctx: str,
        date_str: str,
        timestamp_iso: str,
        chain: CausalChainHandler,
    ) -> Optional[str]:
        """Handles Confluence, Email, or Slack completion for non-eng tickets. Returns completion_id."""

        completion_artifact = ticket.get("completion_artifact", "slack")
        completion_id = None

        if ticket["status"] != "Done":
            if completion_artifact == "confluence" and self._confluence:
                completion_id = self._create_design_doc_stub(
                    author=assignee,
                    participants=[assignee],
                    topic=ticket.get("title", comment_text),
                    ctx=ctx,
                    date_str=date_str,
                    slack_transcript=[],
                )
                if completion_id:
                    chain.append(completion_id)
                    logger.info(
                        f"    [dim]📄 {assignee} completed [{ticket.get('id')}] → {completion_id}[/dim]"
                    )

            elif completion_artifact == "email":
                dept = dept_of_name(assignee, self._org_chart)
                is_sales = "sales" in dept.lower()
                if is_sales and self._crm:
                    completion_id = self._emit_sales_outbound_email(
                        assignee=assignee,
                        ticket=ticket,
                        comment_text=comment_text,
                        date_str=date_str,
                        timestamp=timestamp_iso,
                        chain=chain,
                    )
                else:
                    completion_id = self._emit_completion_email(
                        assignee=assignee,
                        ticket=ticket,
                        comment_text=comment_text,
                        date_str=date_str,
                        timestamp=timestamp_iso,
                    )
                if completion_id:
                    chain.append(completion_id)

            else:
                dept_channel = (
                    dept_of_name(assignee, self._org_chart)
                    .lower()
                    .replace(" ", "-")
                    .replace("_", "-")
                )
                slack_msg = {
                    "user": assignee,
                    "text": (
                        f"Wrapped up [{ticket.get('id')}] {ticket.get('title', '')}. "
                        f"{comment_text}"
                    ),
                    "ts": timestamp_iso,
                    "date": date_str,
                }
                _, completion_id = self._save_slack(
                    [slack_msg], dept_channel, interaction_type="ticket_completion"
                )
                if completion_id:
                    chain.append(completion_id)

            ticket["status"] = "Done"
            ticket["completed_at"] = timestamp_iso

        return completion_id

    def _complete_eng_ticket(
        self,
        ticket: dict,
        assignee: str,
        actor_incident_bound: bool,
        date_str: str,
        timestamp_iso: str,
        chain: CausalChainHandler,
        is_complete: bool,
    ) -> Optional[str]:
        ticket_id = ticket.get("id")
        linked_prs = ticket.get("linked_prs", [])
        ticket_age = self._state.day - ticket.get("in_progress_since", self._state.day)
        actor_clock_ok = self._clock.now(assignee).hour < 17
        actor_clock_iso_format = self._clock.now(assignee).isoformat()
        open_pr_with_changes = bool(linked_prs) and any(
            self._mem._prs.find_one(
                {
                    "pr_id": p,
                    "status": "open",
                    "changes_requested": True,
                    "created_at": {"$lte": actor_clock_iso_format},
                },
                {"_id": 0, "pr_id": 1},
            )
            for p in linked_prs
        )

        spawned_pr_id = None

        force_spawn = (
            ticket_age >= 3
            and not linked_prs
            and not actor_incident_bound
            and not open_pr_with_changes
            and actor_clock_ok
        )

        if (force_spawn or is_complete) and not linked_prs:
            pr = self._git.create_pr(
                author=assignee,
                ticket_id=ticket_id,
                title=f"[{ticket_id}] {ticket['title'][:80]}",
                timestamp=timestamp_iso,
            )
            spawned_pr_id = pr["pr_id"]
            ticket.setdefault("linked_prs", []).append(spawned_pr_id)
            ticket["status"] = "In Review"
            ticket["in_review_since"] = self._state.day
            chain.append(spawned_pr_id)

        return spawned_pr_id

    def _handle_pr_review(
        self,
        eng_plan: EngineerDayPlan,
        item: AgendaItem,
        date_str: str,
    ) -> List[str]:
        """
        Engineer reviews a PR.
        Generates: GitHub review comment thread in Slack #engineering.
        """
        reviewer = eng_plan.name
        pr_id = item.related_id

        pr = self._find_pr(pr_id) or self._find_reviewable_pr(reviewer)
        if not pr:
            return [reviewer]

        author = pr.get("author", reviewer)
        pr_title = pr.get("title", "Unknown PR")
        artifact_time, new_cursor = self._clock.advance_actor(
            author, hours=item.estimated_hrs
        )
        current_actor_time = artifact_time.isoformat()

        ctx = self._mem.context_for_prompt(pr_title, n=2, as_of_time=current_actor_time)
        backstory = persona_utils.get_voice_card(reviewer, "async", self._gd, self._mem)
        p = self._config.get("personas", {}).get(reviewer, {})

        recurrence_hint = ""
        linked_ticket_id = pr.get("linked_ticket") or pr.get("ticket_id", "")
        if linked_ticket_id:
            ticket = self._mem.get_ticket(linked_ticket_id)
            if ticket and ticket.get("recurrence_of"):
                ancestor = self._mem.get_ticket(ticket["recurrence_of"])
                ancestor_root_cause = ancestor.get("root_cause", "") if ancestor else ""
                recurrence_hint = (
                    f"Note: this PR fixes {linked_ticket_id}, which is a recurrence of "
                    f"{ticket['recurrence_of']} ({ticket.get('recurrence_gap_days', '?')} days ago). "
                    f"Prior root cause: {ancestor_root_cause[:120]}"
                )

        prior_reviews = pr.get("comments", [])
        round_count = len(prior_reviews)

        prior_reviews = pr.get("comments", [])
        review_history = ""
        if prior_reviews:
            rounds = "\n".join(
                f"  - {c['author']} ({c['date']}): [{c.get('verdict', '?')}] {c['text'][:120]}"
                for c in prior_reviews[-6:]  # last 6 comments max
            )
            review_history = f"\n--- PRIOR REVIEW ROUNDS ---\n{rounds}\n\n"

        author_persona = self._config.get("personas", {}).get(author, {})
        expertise_list = author_persona.get("expertise", ["general tasks"])
        expertise_str = ", ".join(str(e) for e in expertise_list[:5])
        author_dept = next(
            (d for d, members in self._org_chart.items() if author in members),
            "Unknown",
        )

        reviewer_persona = self._config.get("personas", {}).get(reviewer, {})
        reviewer_expertise_list = reviewer_persona.get("expertise", ["general tasks"])
        reviewer_expertise_str = ", ".join(str(e) for e in reviewer_expertise_list[:5])
        reviewer_dept = next(
            (d for d, members in self._org_chart.items() if reviewer in members),
            "Unknown",
        )

        orphaned_domain_context = ""
        if self._lifecycle:
            all_domains = list(
                self._mem._db["domain_registry"].find({"primary_owner": None})
            )
            pr_title_lower = pr_title.lower()
            for rec in all_domains:
                if any(tag in pr_title_lower for tag in rec.get("system_tags", [])):
                    pct = int(rec.get("documentation_coverage", 0) * 100)
                    known_by = rec.get("known_by", [])
                    orphaned_domain_context += (
                        f"\n⚠ '{rec['domain']}' is an orphaned domain: "
                        f"former owner={rec.get('former_owner', 'unknown')}, "
                        f"documentation={pct}%, "
                        f"partial knowledge: {known_by or 'nobody'}."
                    )

        agent = make_agent(
            role=f"{reviewer} — {p.get('social_role', 'Code Reviewer')}",
            goal=f"Write a PR review comment as {reviewer} would, reflecting your current stress and style.",
            backstory=backstory,
            llm=self._worker,
        )
        if round_count == 0:
            approval_guidance = (
                "This is the first review. Scrutinize carefully — request changes if "
                "there are correctness, safety, or design issues."
            )
        elif round_count == 1:
            approval_guidance = (
                "This is the second review round. The author has had one round of feedback. "
                "Approve if the main issues have been addressed, even if minor things remain. "
                "Only request changes again if a concrete correctness issue is still unresolved."
            )
        else:
            approval_guidance = (
                f"This is review round {round_count + 1}. The author has responded to "
                f"{round_count} rounds of feedback. You MUST approve unless there is an "
                f"obvious unresolved bug or security issue. Do not invent new concerns."
            )

        task = Task(
            description=(
                f"You are {reviewer}. You are reviewing this PR by {author}: {pr_title}\n\n"
                f"{review_history}"
                + (
                    f"DOMAIN CONTEXT:{orphaned_domain_context}\n\n"
                    if orphaned_domain_context
                    else ""
                )
                + f"STEP 1 — DECIDE YOUR VERDICT FIRST:\n"
                f"{approval_guidance}\n"
                f"Choose: 'approved' or 'changes_requested'.\n\n"
                f"STEP 2 — WRITE YOUR COMMENT:\n"
                f"Write 1-3 sentences consistent with your verdict. If approved, acknowledge "
                f"what looks good. If changes_requested, name the specific issue only.\n"
                f"Your tone must reflect your current stress level (see your backstory).\n\n"
                f"STEP 3 — AUTHOR KNOWLEDGE AUDIT (answer objectively, NOT in character):\n"
                f"Your expertise as reviewer: [{reviewer_expertise_str}]\n"
                f"Your department: {reviewer_dept}\n"
                f"{author}'s expertise on record: [{expertise_str}]\n"
                f"{author}'s department: {author_dept}\n"
                f"Based on the PR content and your expertise, assess whether {author} "
                f"demonstrates a knowledge gap in the domains this PR touches.\n"
                f"- In 'topics_beyond_author_expertise', list any technical areas where "
                f"{author}'s implementation, approach, or omissions suggest unfamiliarity.\n"
                f"- In 'hedged_claims', list specific decisions or statements in the PR "
                f"that appear incorrect, naive, or underconfident given the problem domain.\n"
                f"- If {author} deferred or left incomplete any section you know should "
                f"exist, list it in 'deferred_or_incomplete'.\n\n"
                f"Use these criteria:\n"
                f"  author_domain_fit:\n"
                f"    'high'   — PR demonstrates fluency: correct abstractions, aware of edge cases, idiomatic\n"
                f"    'medium' — PR is functional but shows shallow understanding or minor missteps\n"
                f"    'low'    — PR shows clear unfamiliarity: wrong patterns, missing fundamentals, or over-reliance on guesswork\n\n"
                f"  gap_classification:\n"
                f"    'none'     — {author}'s expertise aligns with all domains touched by this PR\n"
                f"    'possible' — PR touches 1-2 domains outside {author}'s expertise but implementation looks adequate\n"
                f"    'likely'   — PR touches domains outside {author}'s expertise AND the implementation shows it\n\n"
                f"Respond ONLY with valid JSON. No preamble, no markdown fences.\n"
                f"{{\n"
                f'  "comment": "your review comment here",\n'
                f'  "verdict": "approved" or "changes_requested",\n'
                f'  "metadata": {{\n'
                f'    "author_domain_fit": "low | medium | high",\n'
                f'    "confidence": "low | medium | high",\n'
                f'    "gap_classification": "none | possible | likely",\n'
                f'    "topics_beyond_author_expertise": ["string"],\n'
                f'    "hedged_claims": ["string"],\n'
                f'    "deferred_or_incomplete": ["string"]\n'
                f"  }}\n"
                f"}}"
            ),
            expected_output="Valid JSON only. No preamble, no markdown fences.",
            agent=agent,
        )

        raw_review = str(
            Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        ).strip()

        try:
            clean_review = raw_review.replace("```json", "").replace("```", "").strip()
            parsed_review = json.loads(clean_review)
            review_text = parsed_review.get("comment", raw_review)
            verdict = parsed_review.get("verdict", "approved")
            if verdict not in ("approved", "changes_requested"):
                verdict = "approved"
            review_metadata = parsed_review.get("metadata", {})
        except (json.JSONDecodeError, AttributeError):
            review_text = raw_review
            verdict = "approved"
            review_metadata = {}

        pr_comment = {
            "author": reviewer,
            "date": date_str,
            "timestamp": current_actor_time,
            "text": review_text,
            "verdict": verdict,
        }
        pr.setdefault("comments", []).append(pr_comment)

        linked_ticket_id = pr.get("linked_ticket") or pr.get("ticket_id", "")
        linked_ticket = (
            self._mem.get_ticket(linked_ticket_id) if linked_ticket_id else None
        )

        if verdict == "approved":
            pr["status"] = "merged"
            if linked_ticket:
                linked_ticket["status"] = "Done"
                linked_ticket["updated_at"] = current_actor_time
                self._save_ticket(linked_ticket)
            self._git.merge_pr(pr.get("pr_id", pr_id))
            self._emit_bot_message(
                "engineering",
                "GitHub Actions",
                f"✅ {reviewer} approved and merged {pr.get('pr_id', pr_id)}: {pr_title[:80]}",
                current_actor_time,
            )
            logger.info(
                f"    [green]✅ {pr.get('pr_id', pr_id)} merged — "
                f"{linked_ticket_id} → Done[/green]"
            )
        else:
            pr["changes_requested"] = True
            if linked_ticket and linked_ticket.get("status") == "In Review":
                linked_ticket["status"] = "In Progress"

                linked_ticket["last_review_requested_day"] = self._state.day
                linked_ticket["updated_at"] = current_actor_time
                self._save_ticket(linked_ticket)
            self._emit_bot_message(
                "engineering",
                "GitHub",
                f"🔄 {reviewer} requested changes on {pr.get('pr_id', pr_id)}: "
                f'"{review_text[:100]}"',
                current_actor_time,
            )
            logger.info(
                f"    [yellow]🔄 {pr.get('pr_id', pr_id)} — changes requested, "
                f"{linked_ticket_id} → In Progress[/yellow]"
            )

        pr_path = f"{self._base}/git/prs/{pr.get('pr_id', pr_id)}.json"
        import os
        import json as _json

        os.makedirs(os.path.dirname(pr_path), exist_ok=True)
        with open(pr_path, "w") as f:
            _json.dump(pr, f, indent=2)
        self._mem.upsert_pr(pr)

        if verdict == "changes_requested":
            actors, reply_thread_id = self._emit_review_reply(
                author,
                reviewer,
                pr.get("pr_id", "PR"),
                review_text,
                date_str,
                current_actor_time,
            )
        else:
            actors = [reviewer, author]
            reply_thread_id = None

        self._gd.record_pr_review(author, [reviewer])

        artifact_ids = {"pr": pr.get("pr_id", pr_id or "")}
        if reply_thread_id:
            artifact_ids["slack_thread"] = reply_thread_id

        causal_facts: dict = {}
        active_inc = next(
            (
                i
                for i in self._state.active_incidents
                if i.ticket_id == linked_ticket_id
            ),
            None,
        )
        if active_inc and getattr(active_inc, "causal_chain", None):
            active_inc.causal_chain.append(pr.get("pr_id", pr_id or ""))
            causal_facts["causal_chain"] = active_inc.causal_chain.snapshot()
        elif linked_ticket_id:
            prior = self._mem._events.find_one(
                {
                    "type": "ticket_progress",
                    "artifact_ids.jira": linked_ticket_id,
                    "facts.causal_chain": {"$exists": True},
                },
                {"facts.causal_chain": 1, "_id": 0},
                sort=[("timestamp", -1)],
            )
            ticket_chain = CausalChainHandler(linked_ticket_id)
            if prior:
                for artifact_id in prior.get("facts", {}).get("causal_chain", []):
                    ticket_chain.append(artifact_id)
            ticket_chain.append(pr.get("pr_id", pr_id or ""))
            causal_facts["causal_chain"] = ticket_chain.snapshot()
            artifact_ids["jira"] = linked_ticket_id

        self._mem.log_event(
            SimEvent(
                type="pr_review",
                timestamp=current_actor_time,
                day=self._state.day,
                date=date_str,
                actors=actors,
                artifact_ids=artifact_ids,
                facts={
                    "reviewer": reviewer,
                    "author": author,
                    "pr_title": pr_title,
                    "review_text": review_text,
                    "verdict": verdict,
                    **causal_facts,
                },
                summary=(
                    f"{reviewer} {'approved and merged' if verdict == 'approved' else 'requested changes on'} "
                    f"{pr.get('pr_id', 'PR')} by {author}."
                ),
                tags=["pr_review", "engineering"],
            )
        )

        if self._vader:
            self._score_and_apply_sentiment(review_text, [reviewer], self._vader)

        if self._lifecycle:
            pr_text = f"{pr_title} {review_text}"
            self._lifecycle.scan_for_knowledge_gaps(
                text=pr_text,
                triggered_by=pr.get("pr_id", pr_id or ""),
                day=self._state.day,
                date_str=date_str,
                state=self._state,
                timestamp=current_actor_time,
            )

        # Fire a structured gap event from reviewer audit metadata — distinct
        # from the embedding scan above so the two signal sources are traceable
        if review_metadata:
            domain_fit = review_metadata.get("author_domain_fit", "high")
            gap_class = review_metadata.get("gap_classification", "none")
            beyond = review_metadata.get("topics_beyond_author_expertise", [])
            hedged = review_metadata.get("hedged_claims", [])
            deferred = review_metadata.get("deferred_or_incomplete", [])

            gap_detected = (
                domain_fit == "low"
                or gap_class == "likely"
                or (gap_class == "possible" and len(beyond) > 0)
            )

            if gap_detected:
                self._mem.log_event(
                    SimEvent(
                        type="knowledge_gap_detected",
                        timestamp=current_actor_time,
                        day=self._state.day,
                        date=date_str,
                        actors=[author],
                        artifact_ids={"pr": pr.get("pr_id", pr_id or "")},
                        facts={
                            "detection_method": "reviewer_audit",
                            "reviewer": reviewer,
                            "author": author,
                            "pr_title": pr_title,
                            "author_domain_fit": domain_fit,
                            "author_expertise": expertise_list,
                            "reviewer_expertise": reviewer_expertise_list,
                            "gap_classification": gap_class,
                            "topics_beyond_author_expertise": beyond,
                            "hedged_claims": hedged,
                            "deferred_or_incomplete": deferred,
                        },
                        summary=(
                            f"Knowledge gap detected via reviewer audit: "
                            f"{author} (expertise: {expertise_str}) submitted PR '{pr_title}' "
                            f"with fit={domain_fit}, gap={gap_class}. "
                            f"Reviewed by {reviewer}."
                        ),
                        tags=["knowledge_gap", "pr_review", "reviewer_audit"],
                    )
                )

        logger.info(
            f"    [dim]🔍 {reviewer} reviewed {pr.get('pr_id', 'PR')} [{verdict}][/dim]"
        )
        return actors

    def _handle_pr_review_for_incident(
        self,
        reviewer: str,
        pr: dict,
        date_str: str,
        timestamp: str,
    ) -> None:
        """
        Generate a PR review comment for an incident PR during the review_pending
        stage.  Unlike _handle_pr_review(), this method accepts the PR document
        and reviewer directly (no agenda item needed) so it can be driven from
        flow._advance_incidents() without touching the planner.

        Side-effects:
          - Appends a comment to the PR document in both MongoDB and on disk.
          - Emits a GitHub bot message in #engineering.
          - Emits an author reply if the comment contains a question.
          - Logs a pr_review SimEvent (consistent with normal-day reviews).
          - Updates the social graph (pr_review edge).
        """
        import os
        import json as _json

        pr_id = pr.get("pr_id", "")
        author = pr.get("author", reviewer)
        pr_title = pr.get("title", "Unknown PR")

        review_hrs = 0.5
        artifact_time, _ = self._clock.advance_actor(reviewer, hours=review_hrs)
        current_actor_time = artifact_time.isoformat()

        recurrence_hint = ""
        linked_ticket_id = pr.get("linked_ticket") or pr.get("ticket_id", "")
        if linked_ticket_id:
            ticket = self._mem.get_ticket(
                linked_ticket_id, as_of_time=current_actor_time
            )
            if ticket and ticket.get("recurrence_of"):
                ancestor = self._mem.get_ticket(
                    ticket["recurrence_of"], as_of_time=current_actor_time
                )
                ancestor_root_cause = ancestor.get("root_cause", "") if ancestor else ""
                recurrence_hint = (
                    f"Note: this PR fixes {linked_ticket_id}, which is a recurrence of "
                    f"{ticket['recurrence_of']} ({ticket.get('recurrence_gap_days', '?')} days ago). "
                    f"Prior root cause: {ancestor_root_cause[:120]}"
                )

        ctx = self._mem.context_for_prompt(pr_title, n=2, as_of_time=current_actor_time)
        backstory = persona_utils.get_voice_card(reviewer, "async", self._gd, self._mem)
        p = self._config.get("personas", {}).get(reviewer, {})

        agent = make_agent(
            role=f"{reviewer} — {p.get('social_role', 'Code Reviewer')}",
            goal=f"Write a PR review comment as {reviewer} would, reflecting your current stress and style.",
            backstory=backstory,
            llm=self._worker,
        )
        task = Task(
            description=(
                f"You are {reviewer}. You are reviewing this incident fix PR by {author}: {pr_title}\n\n"
                f"This is an urgent incident fix — keep the review focused on correctness and "
                f"potential regressions rather than style. Write 1-3 sentences as a GitHub PR "
                f"review comment. Be specific — mention the fix approach, flag any edge case, "
                f"or ask a targeted clarifying question.\n\n"
                f"Output the comment text only. No preamble, no 'Here is my review:'.\n\n"
                f"{recurrence_hint} "
                f"--- CONTEXT ---\n{ctx}"
            ),
            expected_output=(
                f"A plain review comment from {reviewer}, 1-3 sentences. "
                f"No preamble, no labels, no quotes around the output."
            ),
            agent=agent,
        )
        review_text = str(
            Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        ).strip()

        pr_comment = {
            "author": reviewer,
            "date": date_str,
            "timestamp": current_actor_time,
            "text": review_text,
        }
        pr.setdefault("comments", []).append(pr_comment)

        pr_path = f"{self._base}/git/prs/{pr_id}.json"
        os.makedirs(os.path.dirname(pr_path), exist_ok=True)
        with open(pr_path, "w") as f:
            _json.dump(pr, f, indent=2)
        self._mem.upsert_pr(pr)

        self._emit_bot_message(
            "engineering",
            "GitHub",
            f'💬 {reviewer} reviewed {pr_id}: "{review_text[:120]}"',
            current_actor_time,
        )

        reply_thread_id = None
        if "?" in review_text:
            actors, reply_thread_id = self._emit_review_reply(
                author,
                reviewer,
                pr_id,
                review_text,
                date_str,
                current_actor_time,
            )
        else:
            actors = [reviewer, author]

        self._gd.record_pr_review(author, [reviewer])

        artifact_ids: dict = {"pr": pr_id}
        if reply_thread_id:
            artifact_ids["slack_thread"] = reply_thread_id

        linked_ticket_id = pr.get("linked_ticket") or pr.get("ticket_id", "")
        causal_facts: dict = {}
        active_inc = next(
            (
                i
                for i in self._state.active_incidents
                if i.ticket_id == linked_ticket_id
            ),
            None,
        )
        if active_inc and getattr(active_inc, "causal_chain", None):
            active_inc.causal_chain.append(pr_id)
            causal_facts["causal_chain"] = active_inc.causal_chain.snapshot()

            t = self._mem.get_ticket(linked_ticket_id, as_of_time=current_actor_time)
            if t:
                t["causal_chain"] = active_inc.causal_chain.snapshot()
                t["updated_at"] = current_actor_time
                self._save_ticket(t)

        self._mem.log_event(
            SimEvent(
                type="pr_review",
                timestamp=current_actor_time,
                day=self._state.day,
                date=date_str,
                actors=actors,
                artifact_ids=artifact_ids,
                facts={
                    "reviewer": reviewer,
                    "author": author,
                    "pr_title": pr_title,
                    "review_text": review_text,
                    "has_question": "?" in review_text,
                    "incident_review": True,
                    **causal_facts,
                },
                summary=f"{reviewer} reviewed incident PR {pr_id} by {author}.",
                tags=["pr_review", "engineering", "incident"],
            )
        )

        if self._vader:
            self._score_and_apply_sentiment(review_text, [reviewer], self._vader)

        logger.info(f"    [dim]🔍 {reviewer} reviewed incident PR {pr_id}[/dim]")

    def _handle_one_on_one(
        self,
        eng_plan: EngineerDayPlan,
        item: AgendaItem,
        date_str: str,
    ) -> List[str]:
        """
        Engineer has a 1:1 with their lead or a collaborator.
        Generates: DM thread (2-4 messages).
        """
        name = eng_plan.name
        collaborator = next(iter(item.collaborator), None) or self._find_lead_for(name)
        participants = [name, collaborator]
        if not collaborator or collaborator == name:
            return [name]

        meeting_start, meeting_end = self._clock.sync_and_advance(
            participants, hours=item.estimated_hrs
        )
        meeting_time_iso = meeting_start.isoformat()

        ctx = self._mem.context_for_person(
            name=name,
            n=2,
            as_of_time=meeting_time_iso,
        )

        backstory = persona_utils.get_voice_card(
            [name, collaborator], "one_on_one", self._gd, self._mem
        )

        past_convs = self._mem.context_for_person_conversations(
            name=name,
            conv_type="1on1",
            as_of_time=meeting_time_iso,
            n=2,
        )

        if past_convs:
            ctx = f"{ctx}\n\n{past_convs}"

        agents, tasks, speakers, prev_task = [], [], [], None
        turn_speakers = [name, collaborator] * (1 + 1)

        for i, speaker in enumerate(turn_speakers):
            is_last = i == len(turn_speakers) - 1
            other = collaborator if speaker == name else name
            p = self._config.get("personas", {}).get(speaker, {})

            agent = make_agent(
                role=f"{speaker} — {p.get('role', 'Engineer')}",
                goal=f"Have a natural 1:1 DM conversation as {speaker}.",
                backstory=backstory,
                llm=self._worker,
            )

            if i == 0:
                base_desc = (
                    f"You are {speaker}. You are in a private Slack DM with {other}.\n\n"
                    f"Context: {ctx}\n\n"
                    f"Open the conversation. Topics might include workload, sprint decisions, "
                    f"or something personal-professional. Use your typing quirks. "
                    f"1-2 sentences. Format: {speaker}: [message]"
                )
            else:
                base_desc = (
                    f"You are {speaker}. Continue the DM conversation. "
                    f"React to what was just said. Stay in character. "
                    f"Format: {speaker}: [message]"
                )

            desc = (
                self._last_turn_desc(speaker, base_desc, "1on1", other)
                if is_last
                else base_desc
            )

            task = Task(
                description=desc,
                expected_output=(
                    "JSON with 'message' and 'summary' keys."
                    if is_last
                    else f"One message from {speaker} in format: {speaker}: [message]"
                ),
                agent=agent,
                context=[prev_task] if prev_task else [],
            )
            agents.append(agent)
            tasks.append(task)
            speakers.append(speaker)
            prev_task = task

        Crew(
            agents=agents, tasks=tasks, process=Process.sequential, verbose=False
        ).kickoff()

        messages = []
        conversation_summary = None
        current_msg_time = datetime.fromisoformat(meeting_time_iso)

        for idx, (speaker, task) in enumerate(zip(speakers, tasks)):
            is_last = idx == len(speakers) - 1
            raw = (task.output.raw or "").strip() if task.output else ""

            if is_last:
                text, conversation_summary = self._extract_last_turn(raw, speaker)
            else:
                text = raw

            if text.lower().startswith(f"{speaker.lower()}:"):
                text = text[len(speaker) + 1 :].strip()
            if text:
                messages.append(
                    {"user": speaker, "text": text, "ts": current_msg_time.isoformat()}
                )
                current_msg_time += timedelta(minutes=random.randint(1, 4))

        p1, p2 = sorted([name, collaborator])
        channel = f"dm_{p1.lower()}_{p2.lower()}"
        slack_path, thread_id = self._save_slack(
            messages, channel, interaction_type="1on1"
        )

        if conversation_summary:
            self._mem.save_conversation_summary(
                conv_type="1on1",
                participants=[name, collaborator],
                summary=conversation_summary,
                day=self._state.day,
                date=date_str,
                timestamp=meeting_time_iso,
                slack_thread_id=thread_id,
                extra_facts={"related_ticket": item.related_id}
                if item.related_id
                else {},
            )

        self._mem.log_event(
            SimEvent(
                type="1on1",
                timestamp=meeting_time_iso,
                day=self._state.day,
                date=date_str,
                actors=[name, collaborator],
                artifact_ids={"slack_path": slack_path, "slack_thread": thread_id},
                facts={
                    "participants": [name, collaborator],
                    "message_count": len(messages),
                },
                summary=f"1:1 between {name} and {collaborator}.",
                tags=["1on1", "slack"],
            )
        )

        if self._vader and messages:
            full_text = " ".join(m["text"] for m in messages)
            self._score_and_apply_sentiment(
                full_text, [name, collaborator], self._vader
            )

        self._gd.record_slack_interaction([name, collaborator])
        logger.info(f"    [dim]👥 1:1 {name} ↔ {collaborator}[/dim]")
        return [name, collaborator]

    def _handle_async_question(
        self,
        eng_plan: EngineerDayPlan,
        item: AgendaItem,
        dept_plan: DepartmentDayPlan,
        date_str: str,
    ) -> List[str]:
        """
        Engineer asks a question in a channel.
        Generates: Slack thread with 2-4 replies from colleagues.
        Uses a single-shot JSON generation to save output tokens while preserving personas.
        """
        asker = eng_plan.name
        collaborator = next(iter(item.collaborator), None) or self._closest_colleague(
            asker
        )

        initial_participants = [asker]
        if collaborator:
            initial_participants.append(collaborator)

        depts = {dept_of_name(p, self._org_chart) for p in initial_participants}
        if len(depts) > 1:
            channel = "digital-hq"
        else:
            channel = dept_of_name(asker, self._org_chart).lower().replace(" ", "-")

        chat_duration_mins = random.randint(5, 45)
        chat_duration_hours = chat_duration_mins / 60.0
        provisional_start, _ = self._clock.sync_and_advance(
            initial_participants, hours=chat_duration_hours
        )
        meeting_time_iso = provisional_start.isoformat()

        ticket_id = item.related_id
        ticket = self._find_ticket(ticket_id, meeting_time_iso)
        ticket_title = ticket["title"] if ticket else item.description

        seed = [collaborator] if collaborator else []
        all_actors = self._expertise_matched_participants(
            topic=ticket_title,
            seed_participants=[asker] + seed,
            as_of_time=meeting_time_iso,
            max_extras=1,
        )

        meeting_start, _ = self._clock.sync_and_advance(all_actors, hours=0)
        meeting_time_iso = meeting_start.isoformat()

        ctx = self._mem.context_for_ticket(
            ticket_id=ticket_id, as_of_time=meeting_time_iso
        )

        relevant_experts = self._mem.find_confluence_experts(
            topic=ticket_title,
            score_threshold=0.75,
            n=3,
            as_of_time=meeting_time_iso,
        )
        doc_hint = (
            "Note: the following internal documentation exists and may be "
            "referenced naturally in this conversation:\n"
            + "\n".join(
                f"  - '{e['title']}' (written by {e['author']}, day {e['day']})"
                for e in relevant_experts
            )
            if relevant_experts
            else ""
        )

        discussions = self._mem.design_discussions_for_ticket(
            ticket_id=ticket_id or "",
            actors=all_actors,
            as_of_time=meeting_time_iso,
            n=2,
        )
        design_hint = self._mem.format_design_discussions_hint(discussions)

        backstory = persona_utils.get_voice_card(
            all_actors, "async", self._gd, self._mem
        )

        responders = [a for a in all_actors if a != asker]
        turn_speakers = [asker] + responders
        if random.random() > 0.5 and responders:
            turn_speakers.append(asker)
        speaker_sequence = ", ".join(turn_speakers)

        combined_hint = f"{doc_hint}\n\n{design_hint}" if design_hint else doc_hint

        tech_stack = self._mem.tech_stack_for_prompt()

        agent = make_agent(
            role="Slack Conversation Simulator",
            goal=(
                "Write a realistic casual Slack Q&A thread between coworkers. "
                "Treat the provided backstory as character reference sheets for the actors you are writing for."
            ),
            backstory=backstory,
            llm=self._worker,
        )

        task = Task(
            description=(
                f"COMPANY CONTEXT: {self._company} which {COMPANY_DESCRIPTION}\n"
                f"Write a full Slack thread where a colleague asks a question.\n\n"
                f"Topic: {ticket_title}\n"
                f"Tech Stack: {tech_stack}\n"
                f"Relevant context: {ctx}\n"
                f"{combined_hint}\n\n"
                f"Turn order: {speaker_sequence}\n\n"
                f"Rules:\n"
                f"- {asker} must open the thread by stating what they are stuck on.\n"
                f"- Responders reply to what came before, ask clarifying questions, or suggest docs.\n"
                f"- CRITICAL: DO NOT use generic corporate openers like 'Hey [Name], could you clarify...', 'Need clarification on...', or 'Hi [Name], could you share...'. Drop right into the question or state a broken assumption.\n"
                f"- Each message must sound distinctly like that person based on their voice card.\n"
                f"- Each message 1-3 sentences max. Do not add narration.\n\n"
                f"Respond ONLY with a JSON array. No preamble, no markdown fences.\n"
                f'[{{"speaker": "Name", "message": "text"}}, ...]'
            ),
            expected_output='A JSON array of objects with "speaker" and "message" keys.',
            agent=agent,
        )

        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())

        turns = _parse_turn_list(raw, "handle_async_question")

        if not turns:
            logger.warning(
                "[async_question] Parsed turns are empty, falling back to empty thread."
            )
            turns = []

        messages = []
        current_msg_time = datetime.fromisoformat(meeting_time_iso)
        for turn in turns:
            speaker = turn.get("speaker", "").strip()
            text = turn.get("message", "").strip()
            if speaker and text:
                messages.append(
                    {"user": speaker, "text": text, "ts": current_msg_time.isoformat()}
                )
                current_msg_time += timedelta(minutes=random.randint(1, 4))

        if not messages:
            return all_actors

        slack_path, thread_id = self._save_slack(
            messages, channel, interaction_type="async_question"
        )

        active_inc = None
        if ticket_id:
            active_inc = next(
                (i for i in self._state.active_incidents if i.ticket_id == ticket_id),
                None,
            )
            if active_inc and getattr(active_inc, "causal_chain", None):
                active_inc.causal_chain.append(thread_id)

        facts = {
            "asker": asker,
            "channel": channel,
            "topic": ticket_title,
            "responders": [a for a in all_actors if a != asker],
            "message_count": len(messages),
        }

        chain = CausalChainHandler(thread_id)
        if ticket_id:
            chain.append(ticket_id)
            prior = self._mem._events.find_one(
                {
                    "type": "ticket_progress",
                    "artifact_ids.jira": ticket_id,
                    "facts.causal_chain": {"$exists": True},
                    "timestamp": {"$lte": meeting_time_iso},
                },
                {"facts.causal_chain": 1, "_id": 0},
                sort=[("timestamp", -1)],
            )
            if prior:
                for aid in prior.get("facts", {}).get("causal_chain", []):
                    chain.append(aid)

        facts["causal_chain"] = chain.snapshot()

        self._mem.log_event(
            SimEvent(
                type="async_question",
                timestamp=meeting_time_iso,
                day=self._state.day,
                date=date_str,
                actors=all_actors,
                artifact_ids={
                    "slack": slack_path,
                    "slack_thread": thread_id,
                    "jira": ticket_id or "",
                },
                facts=facts,
                summary=f"{asker} asked a question in #{channel} about {ticket_title[:50]}.",
                tags=["async_question", "slack"],
            )
        )

        if self._vader and messages:
            full_text = " ".join(m["text"] for m in messages)
            self._score_and_apply_sentiment(full_text, all_actors, self._vader)

        if self._lifecycle and messages:
            thread_text = " ".join(m["text"] for m in messages)
            self._assess_async_thread_gap(
                messages=messages,
                topic=ticket_title,
                asker=asker,
                thread_id=thread_id,
                ticket_id=ticket_id,
                date_str=date_str,
                timestamp=meeting_time_iso,
            )

        self._gd.record_slack_interaction(all_actors)
        logger.info(f"    [dim]❓ {asker} → #{channel} ({len(messages)} msgs)[/dim]")
        return all_actors

    def _handle_design_discussion(
        self,
        eng_plan: EngineerDayPlan,
        item: AgendaItem,
        dept_plan: DepartmentDayPlan,
        date_str: str,
    ) -> List[str]:
        """
        Small group design discussion — 2-3 engineers.

        Routes on item.meeting_medium:
          "slack" → Slack thread (original path) + optional Confluence stub
          "zoom"  → Zoom transcript (.md) + optional Confluence stub
                    The transcript captures decisions verbatim — the knowledge
                    gap evals want content that never surfaced in Jira/Confluence.

        Both paths share participant resolution, clock advance, and SimEvent
        structure so downstream evals see a uniform event schema.
        """
        initiator = eng_plan.name
        collaborators = item.collaborator or (
            [c] if (c := self._closest_colleague(initiator)) else []
        )
        participants = list({initiator} | set(collaborators))

        chat_duration_mins = random.randint(15, 50)
        chat_duration_hours = chat_duration_mins / 60.0
        provisional_start, _ = self._clock.sync_and_advance(
            participants, hours=chat_duration_hours
        )
        meeting_time_iso = provisional_start.isoformat()

        participants = self._expertise_matched_participants(
            topic=item.description,
            seed_participants=participants,
            as_of_time=meeting_time_iso,
            max_extras=1,
        )

        meeting_start, _ = self._clock.sync_and_advance(participants, hours=0)
        meeting_time_iso = meeting_start.isoformat()

        ctx = self._mem.context_for_ticket(
            ticket_id=item.related_id, as_of_time=meeting_time_iso
        )

        medium = getattr(item, "meeting_medium", "slack")

        if medium == "zoom":
            artifact_path, artifact_id, tags = self._run_zoom_design_discussion(
                initiator=initiator,
                participants=participants,
                topic=item.description,
                ctx=ctx,
                meeting_time_iso=meeting_time_iso,
                date_str=date_str,
            )
            medium_key = "zoom_transcript"
        else:
            artifact_path, artifact_id, tags = self._run_slack_design_discussion(
                initiator=initiator,
                participants=participants,
                topic=item.description,
                ctx=ctx,
                meeting_time_iso=meeting_time_iso,
                date_str=date_str,
            )
            medium_key = "slack_thread"

        conf_id = None
        if random.random() < 0.30 and artifact_id:
            stub_messages = [{"user": initiator, "text": item.description}]
            conf_id = self._create_design_doc_stub(
                initiator, participants, item.description, ctx, date_str, stub_messages
            )

        facts = {
            "topic": item.description,
            "participants": participants,
            "spawned_doc": conf_id is not None,
            "medium": medium,
        }

        artifact_ids = {
            medium_key: artifact_id,
            "artifact_path": artifact_path,
            "confluence": conf_id or "",
        }

        related_ticket_id = item.related_id

        chain = CausalChainHandler(artifact_id)
        if related_ticket_id:
            chain.append(related_ticket_id)
            prior = self._mem._events.find_one(
                {
                    "type": "ticket_progress",
                    "artifact_ids.jira": related_ticket_id,
                    "facts.causal_chain": {"$exists": True},
                    "timestamp": {"$lte": meeting_time_iso},
                },
                {"facts.causal_chain": 1, "_id": 0},
                sort=[("timestamp", -1)],
            )
            if prior:
                for aid in prior.get("facts", {}).get("causal_chain", []):
                    chain.append(aid)
            artifact_ids["jira"] = related_ticket_id

        if conf_id:
            chain.append(conf_id)
        facts["causal_chain"] = chain.snapshot()

        self._mem.log_event(
            SimEvent(
                type="design_discussion",
                timestamp=meeting_time_iso,
                day=self._state.day,
                date=date_str,
                actors=participants,
                artifact_ids=artifact_ids,
                facts=facts,
                summary=(
                    f"{initiator} led {'Zoom' if medium == 'zoom' else 'Slack'} "
                    f"design discussion on '{item.description[:80]}' "
                    f"with {', '.join(p for p in participants if p != initiator)}."
                ),
                tags=tags + (["confluence"] if conf_id else []),
            )
        )

        self._gd.record_slack_interaction(participants)
        logger.info(
            f"    [dim]{'📹' if medium == 'zoom' else '🏗️ '} Design discussion "
            f"[{medium}]: {item.description[:80]} "
            f"({len(participants)} engineers)[/dim]"
        )
        return participants

    def _handle_mentoring(
        self,
        eng_plan: EngineerDayPlan,
        item: AgendaItem,
        date_str: str,
    ) -> List[str]:
        """
        Senior engineer mentors a junior colleague.
        Generates: DM thread. Boosts social graph edge significantly.
        """
        mentor = eng_plan.name
        mentee = next(iter(item.collaborator), None) or self._find_junior_colleague(
            mentor
        )
        if not mentee or mentee == mentor:
            return [mentor]

        participants = [mentor, mentee]

        session_mins = random.randint(30, 90)
        session_hours = session_mins / 60.0
        meeting_start, meeting_end = self._clock.sync_and_advance(
            participants, hours=session_hours
        )
        meeting_time_iso = meeting_start.isoformat()

        ctx = self._mem.context_for_person(
            name=mentee,
            n=2,
            as_of_time=meeting_time_iso,
        )

        backstory = persona_utils.get_voice_card(
            [mentor, mentee], "mentoring", self._gd, self._mem
        )

        agents, tasks, prev_task = [], [], None
        n_turns = self._turn_count([mentor, mentee], (3, 6))
        speakers = ([mentor, mentee] * ((n_turns // 2) + 1))[:n_turns]
        shared_goal = "Treat the provided backstory as character reference sheets for the actor you are writing for."
        for i, speaker in enumerate(speakers[:n_turns]):
            is_last = i == n_turns - 1
            is_mentor = speaker == mentor

            agent = make_agent(
                role=f"{speaker} — {'Mentor' if is_mentor else 'Mentee'}",
                goal=(
                    f"Guide {mentee} thoughtfully as an experienced engineer. {shared_goal}"
                    if is_mentor
                    else f"Ask genuine questions and absorb guidance as someone still learning. {shared_goal}"
                ),
                backstory=backstory,
                llm=self._worker,
            )
            if i == 0:
                desc = (
                    f"You are {mentor}, opening a mentoring DM with {mentee}.\n\n"
                    f"Context: {ctx}\n\n"
                    f"Start the session — check in, then move toward a topic: career growth, "
                    f"a technical concept, recent work feedback, or navigating a situation. "
                    f"Use your typing quirks. 1-2 sentences. Format: {mentor}: [message]"
                )
            elif is_mentor:
                desc = (
                    f"You are {mentor}. Respond to {mentee}'s message. "
                    f"Be specific — reference real context where you can. "
                    f"Guide, don't lecture. Format: {mentor}: [message]"
                )
            else:
                desc = (
                    f"You are {mentee}. Respond to {mentor}'s guidance. "
                    f"Ask a follow-up, show you're thinking it through, or push back gently "
                    f"if something doesn't make sense. Format: {mentee}: [message]"
                )

            if is_last:
                desc += "\n\nCRITICAL: Since this is the final message, output a JSON object containing 'message' (your response) and 'summary' (a 1-sentence recap of what was discussed)."
                expected_out = "JSON with 'message' and 'summary' keys."
            else:
                expected_out = (
                    f"One message from {speaker} in format: {speaker}: [message]"
                )

            task = Task(
                description=desc,
                expected_output=expected_out,
                agent=agent,
                context=[prev_task] if prev_task else [],
            )
            agents.append(agent)
            tasks.append(task)
            prev_task = task

        Crew(
            agents=agents, tasks=tasks, process=Process.sequential, verbose=False
        ).kickoff()

        messages = []
        conversation_summary = None
        current_msg_time = datetime.fromisoformat(meeting_time_iso)
        for idx, (speaker, task) in enumerate(zip(speakers[:n_turns], tasks)):
            is_last = idx == len(tasks) - 1
            raw = (task.output.raw or "").strip() if task.output else ""

            if is_last:
                text, conversation_summary = self._extract_last_turn(raw, speaker)
            else:
                text = raw

            if text.lower().startswith(f"{speaker.lower()}:"):
                text = text[len(speaker) + 1 :].strip()
            if text:
                messages.append(
                    {"user": speaker, "text": text, "ts": current_msg_time.isoformat()}
                )
                current_msg_time += timedelta(minutes=random.randint(1, 8))

        if not messages:
            return [mentor, mentee]

        p1, p2 = sorted([mentor, mentee])
        channel = f"dm_{p1.lower()}_{p2.lower()}"
        slack_path, thread_id = self._save_slack(
            messages, channel, interaction_type="mentoring"
        )

        if conversation_summary:
            self._mem.save_conversation_summary(
                conv_type="mentoring",
                participants=[mentor, mentee],
                summary=conversation_summary,
                day=self._state.day,
                date=date_str,
                timestamp=meeting_time_iso,
                slack_thread_id=thread_id,
                extra_facts={"related_ticket": item.related_id}
                if item.related_id
                else {},
            )

        # Mentoring is a strong relationship signal
        self._gd.record_slack_interaction([mentor, mentee])
        self._gd.record_slack_interaction([mentor, mentee])  # double boost

        self._mem.log_event(
            SimEvent(
                type="mentoring",
                timestamp=meeting_time_iso,
                day=self._state.day,
                date=date_str,
                actors=[mentor, mentee],
                artifact_ids={"slack_path": slack_path, "slack_thread": thread_id},
                facts={
                    "mentor": mentor,
                    "mentee": mentee,
                    "message_count": len(messages),
                },
                summary=f"{mentor} mentored {mentee}.",
                tags=["mentoring", "slack"],
            )
        )

        logger.info(f"    [dim]🎓 {mentor} → {mentee} (mentoring)[/dim]")
        return [mentor, mentee]

    def _handle_generic_activity(
        self,
        eng_plan: EngineerDayPlan,
        item: AgendaItem,
        date_str: str,
    ) -> List[str]:
        """
        Fallback for unknown activity types — generates a short Slack mention.
        """
        name = eng_plan.name
        channel = dept_of_name(name, self._org_chart).lower().replace(" ", "-")

        cron_time_iso = self._clock.now("system").isoformat()

        self._emit_bot_message(
            channel, name, f"Working on: {item.description}", cron_time_iso
        )
        return [name]

    def _handle_collision_event(self, event: ProposedEvent, date_str: str):
        """Renders the unplanned cross-dept interaction as a Slack thread.
        Uses a single LLM call to generate the full conversation, giving the
        model full arc awareness so tension can escalate or resolve naturally.
        """
        participants = event.actors
        tension = event.facts_hint.get("tension_level", "medium")

        # Sync all participants to a shared start time
        chat_duration_mins = random.randint(5, 30)
        thread_start, _ = self._clock.sync_and_advance(
            participants, hours=chat_duration_mins / 60.0
        )
        thread_start_iso = thread_start.isoformat()

        ctx = self._mem.context_for_prompt(
            event.rationale, n=2, as_of_time=thread_start_iso
        )

        voice_cards = persona_utils.get_voice_card(
            participants, "collision", self._gd, self._mem
        )

        n_turns = {
            "high": random.randint(5, 8),
            "medium": random.randint(4, 6),
            "low": random.randint(3, 5),
        }.get(tension, random.randint(4, 6))

        turn_speakers = [participants[0]] + [
            participants[i % len(participants)] for i in range(1, n_turns)
        ]
        speaker_sequence = ", ".join(turn_speakers)

        tension_guidance = {
            "high": (
                "The exchange should feel genuinely tense — the opener is direct and pressured, "
                "others push back or defend their team, and the thread escalates before any "
                "tentative resolution (or none at all)."
            ),
            "medium": (
                "The exchange is a back-and-forth negotiation — collegial but with real friction. "
                "Each person is trying to get what their team needs while staying professional."
            ),
            "low": (
                "The exchange is collaborative — there's a real problem to solve but no hostility. "
                "It resolves with a clear next step or agreement."
            ),
        }.get(tension, "")

        agent = make_agent(
            role="Slack Conversation Simulator",
            goal="Write a realistic multi-turn Slack exchange between coworkers. Treat the provided backstory as character reference sheets.",
            backstory="You write authentic workplace Slack conversations that reflect each person's distinct voice and the emotional arc of the situation.",
            llm=self._planner,
        )
        task = Task(
            description=(
                f"Write a full Slack thread between coworkers having an unplanned cross-team exchange.\n\n"
                f"Situation: {event.rationale}\n"
                f"Tension level: {tension}\n\n"
                f"{tension_guidance}\n\n"
                f"Participants (voice cards — each person's style, mood, and pet peeves):\n{voice_cards}\n\n"
                f"Context: {ctx}\n\n"
                f"Turn order: {speaker_sequence}\n\n"
                f"Rules:\n"
                f"- Each message must sound distinctly like that person — use their typing quirks\n"
                f"- The conversation must have a natural arc: opening → escalation or negotiation → some resolution or stalemate\n"
                f"- Each message 1-2 sentences\n"
                f"- Do not add narration or stage directions\n\n"
                f"Respond ONLY with a JSON array. No preamble, no markdown fences.\n"
                f'[{{"speaker": "Name", "message": "text"}}, ...]'
            ),
            expected_output='A JSON array of objects with "speaker" and "message" keys, one per turn.',
            agent=agent,
        )
        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()

        turns = _parse_turn_list(raw, "handle_collision_event")
        if not turns:
            logger.warning(
                "[collision] Failed to parse JSON, falling back to empty thread."
            )

        messages = []
        current_msg_time = datetime.fromisoformat(thread_start_iso)
        for turn in turns:
            speaker = turn.get("speaker", "").strip()
            text = turn.get("message", "").strip()
            if speaker and text:
                messages.append(
                    {"user": speaker, "text": text, "ts": current_msg_time.isoformat()}
                )
                current_msg_time += timedelta(minutes=random.randint(1, 4))

        channel = "digital-hq"
        slack_path, thread_id = self._save_slack(messages, channel)

        self._gd.record_slack_interaction(participants)

        self._mem.log_event(
            SimEvent(
                type="org_collision",
                timestamp=thread_start_iso,
                day=self._state.day,
                date=date_str,
                actors=participants,
                artifact_ids={"slack_path": slack_path, "slack_thread": thread_id},
                facts={"tension": tension, "type": event.event_type},
                summary=f"Unplanned {tension} interaction: {event.rationale}",
                tags=["collision", tension],
            )
        )

    def _emit_blocker_slack(
        self,
        asker: str,
        collaborator: str,
        ticket_id: str,
        ticket_title: str,
        blocker_text: str,
        date_str: str,
        timestamp: str,
    ) -> List[str]:
        """
        Short Slack exchange when an engineer is blocked.
        Each participant speaks in their own voice via a dedicated Agent.
        """

        asker_dept = dept_of_name(asker, self._org_chart)
        channel = asker_dept.lower().replace(" ", "-")
        participants = [asker, collaborator]

        backstory = persona_utils.get_voice_card(
            participants, "dm", self._gd, self._mem
        )

        asker_role = (
            self._config.get("personas", {})
            .get(asker, {})
            .get("social_role", "Engineer")
        )
        collab_role = (
            self._config.get("personas", {})
            .get(collaborator, {})
            .get("social_role", "Engineer")
        )

        agent = make_agent(
            role="Slack Conversation Simulator",
            goal="Write a realistic 2-message Slack DM thread where a colleague asks for help with a blocker.",
            backstory=backstory,
            llm=self._worker,
        )

        task = Task(
            description=(
                f"Write a 2-message Slack DM exchange between {asker} ({asker_role}) and {collaborator} ({collab_role}).\n\n"
                f"Context: {asker} is blocked on [{ticket_id}]: {ticket_title}.\n"
                f"Blocker details: {blocker_text[:120]}\n\n"
                f"Rules:\n"
                f"- Turn 1: {asker} reaches out to {collaborator} explaining the blocker. They should sound appropriately stressed/blocked.\n"
                f"- Turn 2: {collaborator} replies naturally—acknowledging, offering help, asking a clarifying question, or redirecting them.\n"
                f"- Treat the provided backstory as strict character reference sheets. They MUST sound like different people and use their typing quirks.\n"
                f"- Each message 1-2 sentences max. No narration.\n\n"
                f"CRITICAL: Respond ONLY with a JSON array containing the two messages. No markdown fences.\n"
                f'[{{"speaker": "Name", "message": "text"}}, {{"speaker": "Name", "message": "text"}}]'
            ),
            expected_output='A JSON array with "speaker" and "message" keys.',
            agent=agent,
        )

        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())

        turns = _parse_turn_list(raw, "handle_blocker")

        messages = []
        current_msg_time = datetime.fromisoformat(timestamp)

        for turn in turns:
            speaker = turn.get("speaker")
            text = turn.get("message", "").strip()
            if text:
                messages.append(
                    {"user": speaker, "text": text, "ts": current_msg_time.isoformat()}
                )
                current_msg_time += timedelta(minutes=random.randint(1, 4))

        if messages:
            slack_path, thread_id = self._save_slack(
                messages, channel, interaction_type="blocker"
            )
            self._gd.record_slack_interaction(participants)

            active_inc = next(
                (i for i in self._state.active_incidents if i.ticket_id == ticket_id),
                None,
            )

            if active_inc and getattr(active_inc, "causal_chain", None):
                active_inc.causal_chain.append(thread_id)
                facts = {
                    "blocker_reason": blocker_text,
                    "causal_chain": active_inc.causal_chain.snapshot(),
                }
            else:
                prior = self._mem._events.find_one(
                    {
                        "type": "ticket_progress",
                        "artifact_ids.jira": ticket_id,
                        "facts.causal_chain": {"$exists": True},
                        "timestamp": {"$lte": timestamp},
                    },
                    {"facts.causal_chain": 1, "_id": 0},
                    sort=[("timestamp", -1)],
                )
                ticket_chain = CausalChainHandler(ticket_id)
                if prior:
                    for artifact_id in prior.get("facts", {}).get("causal_chain", []):
                        ticket_chain.append(artifact_id)
                ticket_chain.append(thread_id)
                facts = {
                    "blocker_reason": blocker_text,
                    "causal_chain": ticket_chain.snapshot(),
                }

            self._mem.log_event(
                SimEvent(
                    type="blocker_flagged",
                    day=self._state.day,
                    date=date_str,
                    timestamp=timestamp,
                    actors=participants,
                    artifact_ids={
                        "slack_path": slack_path,
                        "slack_thread": thread_id,
                        "jira": ticket_id,
                    },
                    facts=facts,
                    summary=f"{asker} is blocked on {ticket_id}, pinged {collaborator}.",
                    tags=["slack", "blocker"],
                )
            )

        logger.info(
            f"    [dim]🚧 Blocker reported: {blocker_text[:80]}... "
            f"({asker} pinged {collaborator})[/dim]"
        )

        return participants

    def _emit_completion_email(
        self,
        assignee: str,
        ticket: dict,
        comment_text: str,
        date_str: str,
        timestamp: str,
    ) -> Optional[str]:
        """
        Generate a short internal completion email for a non-eng ticket.
        Routed to the assignee's department lead as a natural work update.
        Returns a thread/artifact ID for the causal chain, or None on failure.
        """
        ticket_id = ticket.get("id", "")
        ticket_title = ticket.get("title", ticket_id)
        dept = dept_of_name(assignee, self._org_chart)
        lead = self._find_lead_for(assignee) or assignee

        backstory = persona_utils.get_voice_card(assignee, "async", self._gd, self._mem)

        p = self._config.get("personas", {}).get(assignee, {})

        agent = make_agent(
            role=f"{assignee} — {p.get('social_role', 'Engineer')}",
            goal="Write a brief internal email updating your lead on completed work.",
            backstory=backstory,
            llm=self._worker,
        )

        task = Task(
            description=(
                f"You are {assignee}. You just completed ticket [{ticket_id}]: {ticket_title}.\n\n"
                f"Write a short internal email to {lead} (your lead) summarising what you did.\n"
                f"What you did: {comment_text}\n\n"
                f"Rules:\n"
                f"- Subject line must include the ticket ID.\n"
                f"- Body: 2-3 sentences. What was done, any key decision or outcome.\n"
                f"- Sign off with your name.\n"
                f"- Use your typing quirks and current mood.\n\n"
                f"CRITICAL: Respond ONLY with a JSON object containing 'subject' and 'body' keys. No markdown fences, no preamble.\n"
                f"- CRITICAL: Keep the tone accessible. Avoid deep technical jargon, as the recipient may be non-technical.\n"
                f'{{\n  "subject": "Re: [{ticket_id}] ...",\n  "body": "..."\n}}'
            ),
            expected_output='A JSON object with "subject" and "body" keys.',
            agent=agent,
        )

        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())

        clean = raw.replace("```json", "").replace("```", "").strip()

        try:
            email_data = json_repair.loads(clean)

            subject = email_data.get(
                "subject", f"Re: [{ticket_id}] {ticket_title[:60]}"
            )
            body = email_data.get("body", clean)

        except Exception as e:
            logger.warning(f"[email_generation] Failed to parse JSON email: {e}")
            subject = f"Re: [{ticket_id}] Update"
            body = clean

        thread_id = f"email_{ticket_id}_{self._state.day}"

        self._mem.log_event(
            SimEvent(
                type="ticket_completion_email",
                timestamp=timestamp,
                day=self._state.day,
                date=date_str,
                actors=[assignee, lead],
                artifact_ids={"jira": ticket_id, "email_thread": thread_id},
                facts={
                    "ticket_id": ticket_id,
                    "subject": subject,
                    "body": body,
                    "from": assignee,
                    "to": lead,
                    "dept": dept,
                },
                summary=f"{assignee} emailed {lead} re completion of {ticket_id}.",
                tags=["email", "ticket_completion", "non_eng"],
            )
        )

        self._mem.embed_artifact(
            id=thread_id,
            type="email",
            title=subject,
            content=f"From: {assignee}\nTo: {lead}\nSubject: {subject}\n\n{body}",
            day=self._state.day,
            date=date_str,
            timestamp=timestamp,
            metadata={
                "ticket_id": ticket_id,
                "from": assignee,
                "to": lead,
                "dept": dept,
            },
        )

        logger.info(
            f"    [dim]📧 {assignee} → {lead}: completion email for [{ticket_id}][/dim]"
        )
        return thread_id

    def _emit_sales_outbound_email(
        self,
        assignee: str,
        ticket: dict,
        comment_text: str,
        date_str: str,
        timestamp: str,
        chain: "CausalChainHandler",
        opportunity_id: Optional[str] = None,
        account_name: Optional[str] = None,
        contact_name: Optional[str] = None,
        contact_email: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate a customer-facing outbound email for a Sales ticket completion
        or a planner-proposed proactive outreach event.

        Looks up the best matching open SF opportunity for the assignee's account
        if one isn't provided directly. Writes a real .eml, embeds it, calls
        crm.process_outbound_email() to advance the SF opportunity, and logs a
        SimEvent so the causal chain is complete.

        Returns the embed_id (thread_id) or None on failure.
        """
        ticket_id = ticket.get("id", "")
        ticket_title = ticket.get("title", ticket_id)

        if not (account_name and contact_name and contact_email):
            opp = self._crm.get_best_open_opportunity(owner=assignee)
            if opp:
                opportunity_id = opp.get("opportunity_id", opportunity_id)
                account_name = opp.get("account_name", account_name or "the customer")
                contact_name = opp.get("primary_contact", contact_name or account_name)
                contact_email = opp.get(
                    "primary_contact_email",
                    f"{contact_name.lower().replace(' ', '.')}@{account_name.lower().replace(' ', '')}.com",
                )
            else:
                logger.debug(
                    f"[normal_day] No open SF opp for {assignee} — "
                    f"falling back to internal completion email for {ticket_id}"
                )
                return self._emit_completion_email(
                    assignee=assignee,
                    ticket=ticket,
                    comment_text=comment_text,
                    date_str=date_str,
                    timestamp=timestamp,
                )

        p = self._config.get("personas", {}).get(assignee, {})
        backstory = persona_utils.get_voice_card(assignee, "async", self._gd, self._mem)
        stage = opportunity_id and self._crm._sf_o.find_one(
            {"opportunity_id": opportunity_id}, {"stage": 1, "_id": 0}
        )
        stage_label = (
            (stage or {}).get("stage", "active discussion")
            if stage
            else "active discussion"
        )

        agent = make_agent(
            role=f"{assignee} — {p.get('social_role', 'Account Executive')}",
            goal=f"Write a professional outbound email to a customer at {account_name}.",
            backstory=backstory,
            llm=self._worker,
        )
        task = Task(
            description=(
                f"You are {assignee} at {self._company}. You have just completed work on "
                f"ticket [{ticket_id}]: {ticket_title}.\n\n"
                f"What you did: {comment_text}\n\n"
                f"Write a short, professional outbound email to {contact_name} at {account_name}. "
                f"The deal is currently at stage: {stage_label}.\n\n"
                f"The email should naturally reflect the ticket work — e.g. attaching a proposal, "
                f"confirming a renewal quote, following up on a demo, or sharing an update. "
                f"Do NOT mention internal ticket IDs or internal tooling.\n\n"
                f"Rules:\n"
                f"- Subject: relevant, professional, no ticket IDs\n"
                f"- Body: 3-5 sentences. Warm but professional. Sign off with your name and title.\n"
                f"- Match your typing quirks from your backstory.\n\n"
                f"CRITICAL: Respond ONLY with a JSON object with 'subject' and 'body' keys. "
                f"No markdown fences, no preamble.\n"
                f'{{"subject": "...", "body": "..."}}'
            ),
            expected_output='A JSON object with "subject" and "body" keys.',
            agent=agent,
        )
        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())
        clean = raw.replace("```json", "").replace("```", "").strip()

        try:
            email_data = json_repair.loads(clean)
            subject = email_data.get("subject", f"Following up — {account_name}")
            body = email_data.get("body", clean)
            new_stage = email_data.get("crm_stage", stage_label)
        except Exception as exc:
            logger.warning(f"[sales_email] JSON parse failed for {ticket_id}: {exc}")
            subject = f"Following up — {account_name}"
            body = clean

        sender_addr = f"{assignee.lower().replace(' ', '.')}@{self._domain}"
        out_dir = Path(self._base) / "emails" / "outbound" / date_str
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_sender = assignee.lower().replace(" ", "_")
        safe_org = account_name.lower().replace(" ", "_")
        eml_path = out_dir / f"{safe_sender}_to_{safe_org}_{ticket_id}.eml"

        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        msg = MIMEMultipart("alternative")
        msg["From"] = f"{assignee} <{sender_addr}>"
        msg["To"] = f"{contact_name} <{contact_email}>"
        msg["Subject"] = subject
        msg["Date"] = timestamp
        msg["X-OrgForge-Direction"] = "outbound"
        msg.attach(MIMEText(body, "plain"))
        with open(eml_path, "w") as fh:
            fh.write(msg.as_string())

        thread_id = f"sales_email_{ticket_id}_{self._state.day}"

        self._crm.process_outbound_email(
            email_data={
                "sender": assignee,
                "recipient": contact_name,
                "sender_org": self._company,
                "recipient_org": account_name,
                "subject": subject,
                "stage": new_stage,
                "embed_id": thread_id,
            },
            timestamp=timestamp,
            date_str=date_str,
            day=self._state.day,
        )

        self._mem._db["emails"].update_one(
            {"embed_id": thread_id},
            {
                "$setOnInsert": {
                    "embed_id": thread_id,
                    "direction": "outbound",
                    "from_name": assignee,
                    "from_addr": sender_addr,
                    "to_name": contact_name,
                    "to_addr": contact_email,
                    "subject": subject,
                    "body": body,
                    "timestamp": timestamp,
                    "day": self._state.day,
                    "date": date_str,
                    "eml_path": str(eml_path),
                }
            },
            upsert=True,
        )

        chain.append(thread_id)

        self._mem.embed_artifact(
            id=thread_id,
            type="email",
            title=subject,
            content=f"From: {assignee}\nTo: {contact_name} ({account_name})\nSubject: {subject}\n\n{body}",
            day=self._state.day,
            date=date_str,
            timestamp=timestamp,
            metadata={
                "ticket_id": ticket_id,
                "from": assignee,
                "to": contact_name,
                "account": account_name,
                "opportunity_id": opportunity_id,
                "direction": "outbound",
            },
        )

        self._mem.log_event(
            SimEvent(
                type="sales_outbound_email",
                timestamp=timestamp,
                day=self._state.day,
                date=date_str,
                actors=[assignee, contact_name],
                artifact_ids={
                    "jira": ticket_id,
                    "email_thread": thread_id,
                    "eml_path": str(eml_path),
                    "sf_opp": opportunity_id or "",
                },
                facts={
                    "ticket_id": ticket_id,
                    "subject": subject,
                    "from": assignee,
                    "to": contact_name,
                    "account": account_name,
                    "opportunity_id": opportunity_id,
                    "stage": stage_label,
                    "direction": "outbound",
                    "causal_chain": chain.snapshot(),
                },
                summary=f'{assignee} sent outbound email to {contact_name} ({account_name}): "{subject[:80]}"',
                tags=["email", "outbound", "sales", "customer"],
            )
        )

        logger.info(
            f"    [cyan]📤 {assignee} → {contact_name} ({account_name}):[/cyan] {subject[:70]}"
        )
        return thread_id

    def _fire_sales_outreach(self, date_str: str) -> None:
        """
        Deterministic proactive outreach — called once per day after agenda items.

        For each Sales team member, finds their highest-priority open SF opportunity
        that hasn't already been touched by a ticket completion email today, and sends
        one outbound customer email. Fires with ~60% probability per member to avoid
        saturating the corpus every single day.

        No-op if CRM is disabled or the org has no Sales department.
        """
        if not self._crm:
            return

        _OUTREACH_PROB = 0.6

        sales_members = [
            name
            for dept, members in self._org_chart.items()
            if "sales" in dept.lower()
            for name in members
        ]
        if not sales_members:
            return

        already_touched: set = set()
        cooldown_days = 4

        for events in self._mem.get_event_log():
            event_day = getattr(events, "day", -1)
            if (self._state.day - event_day) <= cooldown_days and getattr(
                events, "type", None
            ) == "sales_outbound_email":
                acct = (events.facts or {}).get("account")
                if acct:
                    already_touched.add(acct)

        for sender in sales_members:
            if random.random() > _OUTREACH_PROB:
                continue

            opp = self._crm.get_best_open_opportunity(owner=sender)
            if not opp:
                continue

            account_name = opp.get("account_name", "")
            if account_name in already_touched:
                continue

            acc = self._crm._sf_a.find_one(
                {"name": account_name}, {"_id": 0, "_seq": 0}
            )
            contact_name = (acc or {}).get("primary_contact", account_name)
            contact_email = (acc or {}).get(
                "primary_contact_email",
                f"{contact_name.lower().replace(' ', '.')}@{account_name.lower().replace(' ', '')}.com",
            )

            synthetic_ticket = {
                "id": f"OUTREACH-{self._state.day}-{sender.split()[0].lower()}",
                "title": f"Proactive outreach to {account_name}",
                "status": "Done",
                "completion_artifact": "email",
            }

            chain = CausalChainHandler(root_id=synthetic_ticket["id"])

            opp_updated_str = opp.get("updated_at")
            actor_time = self._clock.now(sender)

            if opp_updated_str:
                opp_updated_time = datetime.fromisoformat(
                    opp_updated_str.replace("Z", "+00:00")
                )

                if actor_time < opp_updated_time:
                    diff_hours = (
                        opp_updated_time - actor_time
                    ).total_seconds() / 3600.0
                    self._clock.advance_actor(sender, hours=(diff_hours + 0.25))
                else:
                    self._clock.advance_actor(sender, hours=0.25)
            else:
                self._clock.advance_actor(sender, hours=0.25)

            timestamp = self._clock.now(sender).isoformat()

            self._emit_sales_outbound_email(
                assignee=sender,
                ticket=synthetic_ticket,
                comment_text=f"Proactive follow-up on open opportunity with {account_name}.",
                date_str=date_str,
                timestamp=timestamp,
                chain=chain,
                opportunity_id=opp.get("opportunity_id"),
                account_name=account_name,
                contact_name=contact_name,
                contact_email=contact_email,
            )
            already_touched.add(account_name)

            self._mem.log_event(
                SimEvent(
                    type="proactive_outreach_initiated",
                    timestamp=self._clock.now(sender).isoformat(),
                    day=self._state.day,
                    date=date_str,
                    actors=[sender],
                    artifact_ids={"synthetic_ticket": synthetic_ticket["id"]},
                    facts={
                        "account": account_name,
                        "opportunity_id": opp.get("opportunity_id"),
                        "causal_chain": chain.snapshot(),
                    },
                    summary=f"{sender} initiated proactive outreach to {account_name}.",
                    tags=["sales", "outreach", "proactive"],
                )
            )

    def _emit_review_reply(
        self,
        author: str,
        reviewer: str,
        pr_id: str,
        review_text: str,
        date_str: str,
        timestamp: str,
    ) -> Tuple[List[str], str]:
        """Author replies to a review question in #engineering."""

        backstory = persona_utils.get_voice_card(author, "async", self._gd, self._mem)
        p = self._config.get("personas", {}).get(author, {})

        agent = make_agent(
            role=f"{author} — {p.get('social_role', 'Engineer')}",
            goal="Reply to a code review question naturally in your own voice.",
            backstory=backstory,
            llm=self._worker,
        )
        task = Task(
            description=(
                f"You are {author}. Reply to this code review comment from {reviewer}.\n\n"
                f"Output format: {author}: [your reply]\n"
                f"Length: 1-2 sentences only. No preamble.\n\n"
                f"Their comment: {review_text[:120]}\n\n"
                f"Answer their question, clarify your intent, or push back if you disagree. Use your typing quirks."
            ),
            expected_output=(
                f"One line only: '{author}: [reply]'. No preamble, no extra lines."
            ),
            agent=agent,
        )
        reply = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()

        thread_id = self._emit_bot_message(
            "engineering",
            "GitHub",
            f'💬 {author} replied to {reviewer}\'s review on {pr_id}: "{reply[:100]}"',
            timestamp=timestamp,
        )
        return [author, reviewer], thread_id

    def _create_design_doc_stub(
        self,
        author: str,
        participants: List[str],
        topic: str,
        ctx: str,  # kept for signature compat, not used
        date_str: str,
        slack_transcript: List[dict],
    ) -> Optional[str]:
        if self._confluence is None:
            logger.warning("[normal_day] No ConfluenceWriter — skipping design doc.")
            return None
        return self._confluence.write_design_doc(
            author=author,
            participants=participants,
            topic=topic,
            slack_transcript=slack_transcript,
            date_str=date_str,
        )

    def _log_deferred_item(self, name: str, item: AgendaItem, date_str: str) -> None:
        """Log a deferred agenda item so the record shows the interruption."""
        current_time_iso = self._clock.now(name).isoformat()

        self._mem.log_event(
            SimEvent(
                type="agenda_item_deferred",
                timestamp=current_time_iso,
                day=self._state.day,
                date=date_str,
                actors=[name],
                artifact_ids={"jira": item.related_id or ""},
                facts={
                    "name": name,
                    "activity_type": item.activity_type,
                    "description": item.description,
                    "defer_reason": item.defer_reason or "unspecified",
                },
                summary=(
                    f"{name}'s '{item.description[:50]}' deferred: "
                    f"{item.defer_reason or 'unspecified'}"
                ),
                tags=["deferred", "agenda"],
            )
        )

    def _log_deep_work(self, name: str, item: AgendaItem, date_str: str) -> None:

        # 1. Deep work takes time! Advance their cursor by the estimated hours.
        # This prevents anyone else from scheduling a 1-on-1 with them during this block.
        artifact_time, new_cursor = self._clock.advance_actor(
            name, hours=item.estimated_hrs
        )

        self._mem.log_event(
            SimEvent(
                type="deep_work_session",
                timestamp=artifact_time.isoformat(),
                day=self._state.day,
                date=date_str,
                actors=[name],
                artifact_ids={},
                facts={"name": name, "focus": item.description},
                summary=f"{name} in deep work: {item.description[:80]}",
                tags=["deep_work"],
            )
        )

    def _maybe_bot_alerts(self) -> None:
        cron_time_iso = self._clock.now("system").isoformat()

        if random.random() < self._config["simulation"].get("aws_alert_prob", 0.4):
            legacy = self._config.get("legacy_system", {})
            self._emit_bot_message(
                "system-alerts",
                "AWS Cost Explorer",
                f"⚠️ Daily budget threshold exceeded. "
                f"{legacy.get('aws_alert_message', 'Cloud costs remain elevated.')}",
                cron_time_iso,
            )
        elif random.random() < self._config["simulation"].get("snyk_alert_prob", 0.2):
            self._emit_bot_message(
                "engineering",
                "Snyk Security",
                "🔒 3 new medium-severity vulnerabilities detected in npm dependencies.",
                cron_time_iso,
            )

    def _maybe_adhoc_confluence(self) -> None:
        if random.random() >= self._config["simulation"].get(
            "adhoc_confluence_prob", 0.3
        ):
            return
        if self._confluence is None:
            return
        # Author and topic are both resolved inside ConfluenceWriter.write_adhoc_page()
        # using daily_active_actors and persona expertise — do not pick randomly here.
        # daily_theme is passed so the topic agent can skew toward operational docs
        # on incident days and strategic docs on calm ones.
        self._confluence.write_adhoc_page()

    def _trigger_watercooler_chat(self, target_actor: str, date_str: str) -> None:
        """Injects non-work chatter, pulling the target actor away from their work."""
        if target_actor not in self._graph:
            return

        edges = self._graph[target_actor]
        if not edges:
            return

        # Pull 1-2 work friends weighted by relationship strength
        colleagues = random.choices(
            list(edges.keys()),
            weights=[edges[n]["weight"] for n in edges.keys()],
            k=random.randint(1, 2),
        )
        participants = list(dict.fromkeys([target_actor] + colleagues))
        if len(participants) < 2:
            return

        chat_duration_mins = random.randint(10, 15)
        thread_start, thread_end = self._clock.sync_and_advance(
            participants, hours=chat_duration_mins / 60.0
        )
        thread_start_iso = thread_start.isoformat()

        # Build topic from participant context
        personas = self._config.get("personas", {})
        participant_interests = []
        for name in participants:
            p = personas.get(name, {})
            interests = p.get("interests", [])
            if interests:
                participant_interests.extend(interests[:2])

        edge_weight = (
            edges.get(colleagues[0], {}).get("weight", 0.5) if colleagues else 0.5
        )
        stress_avg = sum(self._gd._stress.get(n, 30) for n in participants) / len(
            participants
        )
        hour = thread_start.hour

        interests_str = (
            ", ".join(set(participant_interests))
            if participant_interests
            else "general life topics"
        )

        topic_agent = make_agent(
            role="Social Dynamics Observer",
            goal="Pick a realistic watercooler topic for this specific group.",
            backstory="You understand how real coworkers talk based on who they are.",
            llm=self._worker,
        )
        topic_task = Task(
            description=(
                f"Two or more coworkers are taking a break from work at {hour}:00.\n"
                f"Their shared interests include: {interests_str}\n"
                f"Average stress level: {stress_avg:.0f}/100\n"
                f"Relationship closeness (0-20 scale): {edge_weight:.1f}\n\n"
                f"Pick ONE specific, natural watercooler topic for this group. "
                f"High stress → venting or escapism. Low stress → genuine enthusiasm. "
                f"Close colleagues → specific shared references. Acquaintances → generic small talk. "
                f"Pre-lunch hour → food. Friday → weekend. "
                f"Output only the topic as a short phrase. No explanation."
            ),
            expected_output="A short topic phrase, e.g. 'the finale of The Bear' or 'complaining about the new coffee machine'.",
            agent=topic_agent,
        )
        topic = str(
            Crew(agents=[topic_agent], tasks=[topic_task], verbose=False).kickoff()
        ).strip()

        voice_cards = persona_utils.get_voice_card(
            participants, "watercooler", self._gd, self._mem
        )

        speaker_sequence = ", ".join(
            participants[i % len(participants)] for i in range(len(participants) + 1)
        )

        agent = make_agent(
            role="Slack Conversation Simulator",
            goal="Write a realistic casual Slack conversation between coworkers. Treat the provided backstory as character reference sheets.",
            backstory="You write authentic workplace small-talk that reflects each person's distinct personality and current mood.",
            llm=self._worker,
        )
        task = Task(
            description=(
                f"Write a short casual Slack conversation between coworkers chatting about: {topic}\n\n"
                f"Participants (voice cards — each person's typing style and current mood):\n{voice_cards}\n\n"
                f"Turn order: {speaker_sequence}\n\n"
                f"Rules:\n"
                f"- Keep it casual and non-work — this is a distraction, not a meeting\n"
                f"- Each message must sound like that specific person — use their typing quirks and mood\n"
                f"- Messages should feel spontaneous and build on each other naturally\n"
                f"- Each message 1-2 sentences max\n"
                f"- Do not add narration or stage directions\n\n"
                f"Respond ONLY with a JSON array. No preamble, no markdown fences.\n"
                f'[{{"speaker": "Name", "message": "text"}}, ...]'
            ),
            expected_output='A JSON array of objects with "speaker" and "message" keys, one per turn.',
            agent=agent,
        )
        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()

        turns = _parse_turn_list(raw, "trigger_watercooler_chat")
        if not turns:
            logger.warning(
                "[watercooler] Failed to parse JSON, falling back to empty thread."
            )

        messages = []
        current_msg_time = datetime.fromisoformat(thread_start_iso)
        for turn in turns:
            speaker = turn.get("speaker", "").strip()
            text = turn.get("message", "").strip()
            if speaker and text:
                messages.append(
                    {
                        "user": speaker,
                        "text": text,
                        "ts": current_msg_time.isoformat(),
                        "date": date_str,
                    }
                )
                current_msg_time += timedelta(minutes=random.randint(1, 4))

        if messages:
            n = len(participants)
            if n >= 5:
                channel = "random"
            else:
                channel = "dm_" + "_".join(p.lower() for p in sorted(participants))

            slack_path, thread_id = self._mem.log_slack_messages(
                channel=channel,
                messages=messages,
                export_dir=Path(self._base),
            )

            self._gd.record_slack_interaction(participants)

            self._mem.log_event(
                SimEvent(
                    type="watercooler_chat",
                    timestamp=thread_start_iso,
                    day=self._state.day,
                    date=date_str,
                    actors=participants,
                    artifact_ids={"slack_thread": thread_id, "slack_path": slack_path},
                    facts={"topic": topic, "message_count": len(messages)},
                    summary=f"{target_actor} got distracted chatting about {topic} with {len(participants) - 1} others.",
                    tags=["watercooler", "slack", "distraction"],
                )
            )

            logger.info(
                f"    [dim]☕ Distraction: {target_actor} pulled into chat about {topic}[/dim]"
            )

    def _assess_async_thread_gap(
        self,
        messages: List[dict],
        topic: str,
        asker: str,
        thread_id: str,
        ticket_id: Optional[str],
        date_str: str,
        timestamp: str,
    ) -> None:
        """
        Classify whether an async Q&A thread reveals a genuine knowledge gap
        vs a routine question that got answered.

        Uses a fast LLM call (worker model) to classify the thread outcome
        rather than scanning raw text against departed employee embeddings.
        """
        thread_text = "\n".join(f"{m['user']}: {m['text']}" for m in messages)

        asker_persona = self._config.get("personas", {}).get(asker, {})
        asker_expertise = ", ".join(
            str(e) for e in asker_persona.get("expertise", [])[:5]
        )

        agent = make_agent(
            role="Thread Analyst",
            goal="Classify whether a Slack Q&A thread reveals a knowledge gap.",
            backstory="You analyze workplace conversations to identify unresolved questions.",
            llm=self._worker,
        )
        task = Task(
            description=(
                f"Analyze this Slack Q&A thread.\n\n"
                f"Asker: {asker} (expertise: {asker_expertise})\n"
                f"Topic: {topic}\n\n"
                f"Thread:\n{thread_text}\n\n"
                f"Classify the thread outcome:\n"
                f"- 'resolved': the question was answered confidently and correctly\n"
                f"- 'uncertain': responders hedged, guessed, or gave conflicting answers\n"
                f"- 'unresolved': the question went unanswered or was deferred\n"
                f"- 'escalated': someone suggested asking another person or checking docs\n\n"
                f"Respond ONLY with JSON:\n"
                f"{{\n"
                f'  "outcome": "resolved | uncertain | unresolved | escalated",\n'
                f'  "gap_domain": "short topic label if uncertain/unresolved/escalated, '
                f'else empty string",\n'
                f'  "evidence": "one sentence explaining your classification"\n'
                f"}}"
            ),
            expected_output="Valid JSON only.",
            agent=agent,
        )

        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()

        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            outcome = parsed.get("outcome", "resolved")
            gap_domain = parsed.get("gap_domain", "")
            evidence = parsed.get("evidence", "")
        except json.JSONDecodeError:
            return

        if outcome in ("uncertain", "unresolved", "escalated") and gap_domain:
            self._lifecycle.scan_for_knowledge_gaps(
                text=gap_domain,
                triggered_by=thread_id,
                day=self._state.day,
                date_str=date_str,
                state=self._state,
                timestamp=timestamp,
            )

            self._mem.log_event(
                SimEvent(
                    type="knowledge_gap_detected",
                    timestamp=timestamp,
                    day=self._state.day,
                    date=date_str,
                    actors=[m["user"] for m in messages],
                    artifact_ids={
                        "slack_thread": thread_id,
                        "jira": ticket_id or "",
                    },
                    facts={
                        "detection_method": "async_thread_classification",
                        "outcome": outcome,
                        "gap_domain": gap_domain,
                        "evidence": evidence,
                        "asker": asker,
                        "asker_expertise": asker_expertise,
                        "topic": topic,
                    },
                    summary=(
                        f"Async thread {outcome}: {asker} asked about '{topic}' — "
                        f"{evidence}"
                    ),
                    tags=["knowledge_gap", "slack", "async_question"],
                )
            )

    def _last_turn_desc(
        self,
        speaker: str,
        base_desc: str,
        conv_type: str,  # "1on1" | "mentoring"
        other_participant: str,
    ) -> str:
        """
        Wraps the last turn's task description so the LLM outputs JSON with
        both `message` and `summary` fields — saving a second LLM call.

        The `summary` is a 1-2 sentence third-person recap of the whole
        conversation — not the final message.  It's stored in MongoDB for
        future context_for_person_conversations() lookups.

        Args:
            speaker:           The last speaker in the conversation.
            base_desc:         The normal task description for this turn.
            conv_type:         "1on1" or "mentoring" — used in the summary prompt.
            other_participant: The other person in the conversation.

        Returns:
            A modified task description that instructs the LLM to output JSON.

        Usage:
            # Replace the last iteration's desc with:
            if i == len(turn_speakers) - 1:
                desc = self._last_turn_desc(speaker, desc, "1on1", other_name)
        """
        return (
            f"{base_desc}\n\n"
            f"IMPORTANT — For this final message only, respond in JSON (no markdown fences):\n"
            f"{{\n"
            f'  "message": "your reply as {speaker} — same format and length as before",\n'
            f'  "summary": "1-2 sentence third-person recap of the full {conv_type} '
            f"between {speaker} and {other_participant}. "
            f'What was discussed? What was decided or left open? Written for future reference."\n'
            f"}}"
        )

    def _extract_last_turn(self, raw_output: str, speaker: str) -> tuple:
        """
        Parses the last turn's output, which may be JSON (message + summary)
        or plain text (fallback if the LLM didn't follow instructions).

        Returns:
            (message_text: str, summary_text: str | None)

        Usage:
            text, summary = self._extract_last_turn(task.output.raw, speaker)
            if text.lower().startswith(f"{speaker.lower()}:"):
                text = text[len(speaker) + 1:].strip()
        """
        raw = (raw_output or "").strip()
        # Strip accidental markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "message" in parsed:
                message = parsed["message"].strip()
                summary = parsed.get("summary", "").strip() or None
                return message, summary
        except (json.JSONDecodeError, ValueError):
            pass
        # Fallback — treat the whole output as the message
        return raw, None

    def _save_slack(
        self, messages: List[dict], channel: str, interaction_type: str = "general"
    ) -> Tuple[str, str]:
        """Write Slack messages to disk + MongoDB. Returns export path."""
        date_str = str(self._state.current_date.date())
        for m in messages:
            m.setdefault("date", date_str)

        if not all(m.get("is_bot") for m in messages):
            messages = self._threat.inject_slack(
                messages,
                channel=channel,
                day=self._state.day,
                current_date=self._state.current_date,
            )

        slack_path, thread_id = self._mem.log_slack_messages(
            channel=channel, messages=messages, export_dir=Path(self._base)
        )

        if messages:
            # Concatenate the full conversation so the RAG context preserves the flow
            full_transcript = "\n".join(f"{m['user']}: {m['text']}" for m in messages)
            start_timestamp = messages[0].get(
                "ts", self._clock.now("system").isoformat()
            )

            self._mem.embed_artifact(
                id=thread_id,
                type="slack_thread",
                title=f"{interaction_type.replace('_', ' ').title()} in #{channel}",
                content=full_transcript,
                day=self._state.day,
                date=date_str,
                timestamp=start_timestamp,
                metadata={
                    "channel": channel,
                    "interaction_type": interaction_type,
                    "participants": list({m["user"] for m in messages}),
                    "message_count": len(messages),
                },
            )

        return slack_path, thread_id

    def _run_slack_design_discussion(
        self,
        initiator: str,
        participants: List[str],
        topic: str,
        ctx: str,
        meeting_time_iso: str,
        date_str: str,
    ) -> Tuple[str, str, List[str]]:
        """
        Returns (slack_path, thread_id, tags).
        """

        backstory = persona_utils.get_voice_card(
            participants, "async", self._gd, self._mem
        )

        turn_speakers = [initiator] + [
            participants[i % len(participants)] for i in range(1, random.randint(5, 8))
        ]
        speaker_sequence = ", ".join(turn_speakers)

        agent = make_agent(
            role="Slack Conversation Simulator",
            goal=(
                "Write a realistic multi-turn Slack technical design discussion. "
                "Treat the provided backstory as character reference sheets."
            ),
            backstory=backstory,
            llm=self._planner,
        )

        task = Task(
            description=(
                f"COMPANY CONTEXT: {self._company} which {COMPANY_DESCRIPTION}\n"
                f"Write a full Slack thread for a design discussion.\n\n"
                f"Topic: {topic}\n"
                f"Relevant context: {ctx}\n\n"
                f"Turn order: {speaker_sequence}\n\n"
                f"Rules:\n"
                f"- {initiator} opens by framing the problem or trade-off.\n"
                f"- Others raise trade-offs, push back, or propose a next step. Do not just agree.\n"
                f"- CRITICAL: DO NOT use generic openers like 'Hey team, let's discuss...'\n"
                f"- Each message 1-3 sentences max. No narration.\n\n"
                f"Respond ONLY with a JSON array. No preamble, no markdown fences.\n"
                f'[{{"speaker": "Name", "message": "text"}}, ...]'
            ),
            expected_output='JSON array of {{"speaker", "message"}} objects.',
            agent=agent,
        )

        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())
        turns = _parse_turn_list(raw, "_run_slack_design_discussion")

        messages = []
        current_msg_time = datetime.fromisoformat(meeting_time_iso)
        for turn in turns:
            speaker = turn.get("speaker", "").strip()
            text = turn.get("message", "").strip()
            if speaker and text:
                messages.append(
                    {"user": speaker, "text": text, "ts": current_msg_time.isoformat()}
                )
                current_msg_time += timedelta(minutes=random.randint(1, 8))

        depts = {dept_of_name(p, self._org_chart) for p in participants}
        channel = (
            "digital-hq"
            if len(depts) > 1
            else dept_of_name(initiator, self._org_chart).lower().replace(" ", "-")
        )

        slack_path, thread_id = self._save_slack(
            messages, channel, interaction_type="design"
        )
        return slack_path, thread_id, ["design_discussion", "slack"]

    def _run_zoom_design_discussion(
        self,
        initiator: str,
        participants: List[str],
        topic: str,
        ctx: str,
        meeting_time_iso: str,
        date_str: str,
    ) -> Tuple[str, str, List[str]]:
        """
        Zoom-transcript path for design discussions.

        The LLM writes a realistic meeting transcript — attendees speak in full
        sentences, decisions are stated explicitly, action items are called out.
        This is the knowledge-gap surface: decisions made verbally that won't
        appear in Jira or Confluence unless someone writes them up afterward.

        Returns (file_path, transcript_id, tags).
        """

        backstory = persona_utils.get_voice_card(
            participants, "sync", self._gd, self._mem
        )

        agent = make_agent(
            role="Meeting Transcript Generator",
            goal=(
                "Write a realistic Zoom meeting transcript for a technical design session. "
                "Treat the provided backstory as character reference sheets."
            ),
            backstory=backstory,
            llm=self._planner,
        )

        attendee_list = ", ".join(participants)

        task = Task(
            description=(
                f"COMPANY CONTEXT: {self._company} which {COMPANY_DESCRIPTION}\n\n"
                f"Write a realistic Zoom meeting transcript for a live design discussion.\n\n"
                f"Topic: {topic}\n"
                f"Attendees: {attendee_list}\n"
                f"Host/initiator: {initiator}\n"
                f"Relevant context: {ctx}\n\n"
                f"## Format rules\n"
                f"- Output a JSON array. Each element is one speaker turn.\n"
                f"- Schema: [{{'speaker': 'Name', 'message': 'text'}}]\n"
                f"- Turns should be 1-4 sentences. People interrupt, trail off, agree, disagree.\n"
                f"- {initiator} opens by stating the meeting goal clearly.\n"
                f"- The group must reach at least one explicit decision or action item before the meeting ends.\n"
                f"- Include a brief wrap-up turn from {initiator} that states the decision and who owns the follow-up.\n"
                f"- DO NOT use filler like 'Great point!' or 'Absolutely!' as standalone turns.\n"
                f"- Speak naturally — 'gonna', contractions, occasional 'um' are fine.\n"
                f"- NO narration, NO stage directions, NO markdown in message text.\n\n"
                f"Respond ONLY with the JSON array. No preamble, no markdown fences."
            ),
            expected_output='JSON array of {{"speaker", "message"}} objects only.',
            agent=agent,
        )

        raw = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())
        turns = _parse_turn_list(raw, "_run_zoom_design_discussion")

        if not turns:
            logger.warning(
                "[zoom_design_discussion] Empty transcript — falling back to Slack path."
            )
            return self._run_slack_design_discussion(
                initiator, participants, topic, ctx, meeting_time_iso, date_str
            )

        transcript_path, transcript_id = self._save_zoom_transcript(
            turns=turns,
            participants=participants,
            topic=topic,
            meeting_time_iso=meeting_time_iso,
            date_str=date_str,
        )
        return (
            transcript_path,
            transcript_id,
            ["design_discussion", "zoom", "transcript"],
        )

    def _save_zoom_transcript(
        self,
        turns: List[dict],
        participants: List[str],
        topic: str,
        meeting_time_iso: str,
        date_str: str,
    ) -> Tuple[str, str]:
        """
        Persists a Zoom-style meeting transcript.

        Storage:
          - Disk:    {output_dir}/zoom/{date}/zoom_{id}.md   (human-readable)
          - MongoDB: embed_artifact type="zoom_transcript"
                     (same RAG surface as slack_thread / confluence)

        Returns (file_path, transcript_id).
        """
        import os
        import uuid

        transcript_id = f"zoom_{date_str}_{uuid.uuid4().hex[:8]}"

        lines = [
            "# Zoom Meeting Transcript",
            f"**Date:** {date_str}",
            f"**Topic:** {topic}",
            f"**Attendees:** {', '.join(participants)}",
            "",
            "---",
            "",
        ]

        current_ts = datetime.fromisoformat(meeting_time_iso)
        full_text_parts = []

        for turn in turns:
            speaker = turn.get("speaker", "").strip()
            message = turn.get("message", "").strip()
            if not (speaker and message):
                continue
            ts_label = current_ts.strftime("%H:%M:%S")
            lines.append(f"**[{ts_label}] {speaker}:** {message}")
            lines.append("")
            full_text_parts.append(f"{speaker}: {message}")
            current_ts += timedelta(minutes=random.randint(1, 4))

        md_content = "\n".join(lines)
        full_text = "\n".join(full_text_parts)

        zoom_dir = os.path.join(self._base, "zoom", date_str)
        os.makedirs(zoom_dir, exist_ok=True)
        file_path = os.path.join(zoom_dir, f"{transcript_id}.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        embed_metadata = {
            "participants": participants,
            "topic": topic,
            "turn_count": len(turns),
            "medium": "zoom",
        }

        self._mem.embed_artifact(
            id=transcript_id,
            type="zoom_transcript",
            title=f"Zoom: {topic[:80]}",
            content=full_text,
            day=self._state.day,
            date=date_str,
            timestamp=meeting_time_iso,
            metadata=embed_metadata,
        )

        logger.info(
            f"    [dim]📹 Zoom transcript saved: {transcript_id} "
            f"({len(turns)} turns, {len(participants)} attendees)[/dim]"
        )
        return file_path, transcript_id

    def _save_md(self, path: str, content: str) -> None:
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def _save_ticket(self, ticket: dict) -> None:
        import os
        import json as _json

        path = f"{self._base}/jira/{ticket['id']}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            _json.dump(ticket, f, indent=2)

        self._mem.upsert_ticket(ticket)

    def _emit_bot_message(
        self, channel: str, bot_name: str, text: str, timestamp: str
    ) -> str:
        """Unified 4-arg signature matching flow.py._emit_bot_message."""
        date_str = str(self._state.current_date.date())
        msg = {
            "user": bot_name,
            "email": f"{bot_name.lower()}@bot.{self._domain}",
            "text": text,
            "ts": timestamp,
            "date": date_str,
            "is_bot": True,
        }

        _, thread_id = self._save_slack(
            messages=[msg],
            channel=channel,
            interaction_type="bot_message",
        )
        return thread_id

    def _find_ticket(
        self, ticket_id: Optional[str] = "", meeting_time_iso: Optional[str] = ""
    ) -> Optional[dict]:
        if not ticket_id and meeting_time_iso:
            return None
        return self._mem.get_ticket(ticket_id, as_of_time=meeting_time_iso)

    def _find_pr(
        self, pr_id: Optional[str], timestamp: Optional[str] = ""
    ) -> Optional[dict]:
        if not pr_id:
            return None
        return self._mem._prs.find_one(
            {"pr_id": pr_id, "created_at": {"$lte": timestamp}}, {"_id": 0}
        )

    def _find_reviewable_pr(self, reviewer: str) -> Optional[dict]:
        """Find an open PR where this person is listed as a reviewer."""
        prs = self._mem.get_reviewable_prs_for(reviewer)
        return random.choice(prs) if prs else None

    def _closest_colleague(self, name: str) -> Optional[str]:
        """Returns the highest-weight neighbour in the social graph."""
        if name not in self._graph:
            return None
        neighbors = [
            (n, self._graph[name][n].get("weight", 0))
            for n in self._graph.neighbors(name)
            if n in self._all_names
        ]
        if not neighbors:
            return None
        return max(neighbors, key=lambda x: x[1])[0]

    def _find_lead_for(self, name: str) -> Optional[str]:
        dept = dept_of_name(name, self._org_chart)
        leads = self._config.get("leads", {})
        return leads.get(dept)

    def _find_junior_colleague(self, senior: str) -> Optional[str]:
        """Find a colleague with lower tenure — crude proxy for junior status."""
        dept = dept_of_name(senior, self._org_chart)
        members = self._org_chart.get(dept, [])
        personas = self._config.get("personas", {})
        senior_tenure = personas.get(senior, {}).get("tenure", "mid")

        # Tenure ordering: intern < junior < mid < senior < staff < principal
        _RANK = {
            "intern": 0,
            "junior": 1,
            "mid": 2,
            "senior": 3,
            "staff": 4,
            "principal": 5,
        }
        senior_rank = _RANK.get(str(senior_tenure).lower().split()[0], 2)

        juniors = [
            n
            for n in members
            if n != senior
            and _RANK.get(
                str(personas.get(n, {}).get("tenure", "mid")).lower().split()[0], 2
            )
            < senior_rank
        ]
        return random.choice(juniors) if juniors else None

    def _channel_members(self, channel: str, exclude: str) -> List[str]:
        """Returns likely members of a channel based on dept name."""
        for dept, members in self._org_chart.items():
            if dept.lower().replace(" ", "-") == channel:
                return [n for n in members if n != exclude]
        return [n for n in self._all_names if n != exclude]

    def graph_dynamics_record(self, participants: List[str]) -> None:
        self._gd.record_slack_interaction(participants)

    def _expertise_matched_participants(
        self,
        topic: str,
        seed_participants: List[str],
        as_of_time: Optional[str] = None,
        max_extras: int = 2,
    ) -> List[str]:
        """
        Given a topic string and a seed participant list, return an augmented
        list that pulls in people whose persona expertise overlaps the topic.

        Priority order:
          1. Anyone in seed_participants stays.
          2. Authors of semantically similar Confluence pages already in MongoDB
             are injected as subject-matter experts.  This uses vector similarity
             via Memory.find_confluence_experts() -- no new embed calls are made
             for stored pages, only one embed call for the topic query string.
             Causal ordering is enforced by the as_of_time cutoff so a page
             being written right now cannot be referenced before it is saved.
          3. Up to max_extras additional people whose persona expertise tags
             appear in the topic string, weighted by social-graph proximity to
             the seed so the conversation stays socially plausible.

        People with zero expertise overlap are never added -- primary eval guard
        against off-domain participants joining technical threads.
        """
        topic_lower = topic.lower()
        participants: List[str] = list(seed_participants)

        # 1. Semantic expert injection via MongoDB vector search.
        #    find_confluence_experts() reuses already-stored embeddings, so the
        #    only new embed call is for the topic query string itself.
        #    as_of_time enforces causal ordering at sub-day precision.
        experts = self._mem.find_confluence_experts(
            topic=topic,
            score_threshold=0.75,
            n=5,
            as_of_time=as_of_time,
        )
        for e in experts:
            author = e.get("author")
            if author and author in self._all_names and author not in participants:
                participants.append(author)

        # 2. Expertise-tag fallback for engineers with no Confluence history yet
        #    (new hires, or topics that haven't been documented before).
        if len(participants) >= len(seed_participants) + max_extras:
            return participants

        candidates: List[tuple] = []
        for name in self._all_names:
            if name in participants:
                continue
            persona = self._config.get("personas", {}).get(name, {})
            expertise = [e.lower() for e in persona.get("expertise", [])]
            hits = sum(1 for tag in expertise if tag in topic_lower)
            if hits == 0:
                continue
            graph_weight = max(
                (
                    self._graph[name][p].get("weight", 0.0)
                    for p in seed_participants
                    if self._graph.has_edge(name, p)
                ),
                default=0.0,
            )
            candidates.append((name, hits + graph_weight))

        candidates.sort(key=lambda x: x[1], reverse=True)
        for name, _ in candidates[:max_extras]:
            if name not in participants:
                participants.append(name)

        return participants

    def _score_and_apply_sentiment(
        self,
        text: str,
        actors: List[str],
        vader,
    ) -> float:
        """Score text sentiment and apply stress nudge to involved actors."""
        compound = vader.polarity_scores(text)["compound"]
        self._gd.apply_sentiment_stress(actors, compound)
        return compound

    def _turn_count(self, participants: List[str], default_range: tuple) -> int:
        """
        Returns a turn count inversely scaled to average participant stress.
        High stress → shorter exchange. Low stress → fuller conversation.
        """
        avg_stress = sum(self._gd._stress.get(n, 30) for n in participants) / len(
            participants
        )

        if avg_stress > 80:
            return default_range[0]  # floor — terse, get-it-done
        elif avg_stress > 60:
            return random.randint(*default_range[:2])  # low end of range
        else:
            return random.randint(*default_range)  # full range


def dept_of_name(name: str, org_chart: Dict[str, List[str]]) -> str:
    for dept, members in org_chart.items():
        if name in members:
            return dept
    return "Unknown"


TURN_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"speaker": {"type": "string"}, "message": {"type": "string"}},
        "required": ["speaker", "message"],
    },
}


def _parse_turn_list(raw: str, caller: str) -> list:
    """
    Robustly extract a JSON array of turn dicts from an LLM response
    using schema-guided repair.
    """
    clean = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json_repair.loads(
            clean, schema=TURN_SCHEMA, schema_repair_mode="salvage"
        )

        if isinstance(parsed, list):
            return parsed

        return []

    except ValueError as e:
        logger.warning(
            f"[{caller}] LLM output completely failed schema validation: {e}"
        )
        return []
    except Exception as e:
        logger.warning(f"[{caller}] Unexpected parsing failure: {e}")
        return []
