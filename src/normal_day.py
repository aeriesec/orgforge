from __future__ import annotations

import json
import logging
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from crewai import Agent, Task, Crew

from memory import Memory, SimEvent
from graph_dynamics import GraphDynamics
from planner_models import (
    AgendaItem,
    DepartmentDayPlan,
    EngineerDayPlan,
    OrgDayPlan,
    ProposedEvent,
)

logger = logging.getLogger("orgforge.normalday")


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITY HANDLERS
# One method per agenda item activity_type.
# Each returns a list of artifacts written and actors involved.
# ─────────────────────────────────────────────────────────────────────────────

class NormalDayHandler:

    def __init__(
        self,
        config,
        mem:            Memory,
        state,
        graph_dynamics: GraphDynamics,
        social_graph,
        git,
        worker_llm,
        planner_llm,
        clock,
        persona_helper,
        confluence_writer=None
    ):
        self._config         = config
        self._mem            = mem
        self._state          = state
        self._gd             = graph_dynamics
        self._graph          = social_graph
        self._git            = git
        self._worker         = worker_llm
        self._planner        = planner_llm
        self._base           = config["simulation"].get("output_dir", "./export")
        self._domain         = config["simulation"]["domain"]
        self._company        = config["simulation"]["company_name"]
        self._all_names      = [n for dept in config["org_chart"].values() for n in dept]
        self._org_chart      = config["org_chart"]
        self._clock          = clock
        self._persona_helper = persona_helper
        self._confluence     = confluence_writer
        self._registry       = getattr(confluence_writer, "_registry", None)

    # ─── PUBLIC ENTRY POINT ───────────────────────────────────────────────────

    def handle(self, org_plan: Optional[OrgDayPlan]) -> None:
        """Processes both planned agenda items and unplanned org collisions."""
        logger.info(f"  [bold blue]💬 Normal Day Activity[/bold blue]")
        date_str = str(self._state.current_date.date())

        if org_plan is None:
            self._legacy_slack_chatter(date_str)
            return

        # 1. Execute the planned daily work
        self._execute_agenda_items(org_plan, date_str)

        # 2. Execute the unplanned cross-dept collisions (Synergy or Friction)
        for event in org_plan.collision_events:
            self._handle_collision_event(event, date_str)

        # These fire regardless — they're ambient signals, not agenda-driven
        self._maybe_bot_alerts()
        self._maybe_adhoc_confluence()

    # ─── AGENDA EXECUTION ─────────────────────────────────────────────────────

    def _execute_agenda_items(self, org_plan: OrgDayPlan, date_str: str) -> None:
        """
        Walk every engineer's agenda across all departments.
        Dispatch each non-deferred item to the appropriate activity handler.
        Deferred items are logged as SimEvents so the record shows the interruption.
        """
        all_participants: List[str] = []
    
        for dept, dept_plan in org_plan.dept_plans.items():
            for eng_plan in dept_plan.engineer_plans:
                
                # Per-engineer-per-day gate: evaluated once, fires at most once
                watercooler_prob = self._config["simulation"].get("watercooler_prob", 0.15)
                will_be_distracted = random.random() < watercooler_prob
                distraction_fired = False
                
                # Pick a random non-deferred item to attach the distraction to
                non_deferred_indices = [
                    idx for idx, item in enumerate(eng_plan.agenda) if not item.deferred
                ]
                distraction_index = (
                    random.choice(non_deferred_indices) if non_deferred_indices else None
                )
                
                for idx, item in enumerate(eng_plan.agenda):
                    if item.deferred:
                        self._log_deferred_item(eng_plan.name, item, date_str)
                        continue
                    
                    if will_be_distracted and not distraction_fired and idx == distraction_index:
                        self._trigger_watercooler_chat(eng_plan.name, date_str)
                        penalty_hours = random.uniform(0.16, 0.25)
                        item.estimated_hrs += penalty_hours
                        self._clock.advance_actor(eng_plan.name, penalty_hours)
                        distraction_fired = True

                    participants = self._dispatch(eng_plan, item, dept_plan, date_str)
                    all_participants.extend(participants)

        if all_participants:
            unique = list(set(all_participants))
            self.graph_dynamics_record(unique)

    def _dispatch(
        self,
        eng_plan:  EngineerDayPlan,
        item:      AgendaItem,
        dept_plan: DepartmentDayPlan,
        date_str:  str,
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
            # Deep work is intentionally silent — no artifact, no Slack
            # But we log a SimEvent so the day_summary knows who was heads-down
            self._log_deep_work(eng_plan.name, item, date_str)
            return [eng_plan.name]
        elif t == "code_review_comment":
            return self._handle_pr_review(eng_plan, item, date_str)
        else:
            # Unknown activity type — generate a generic Slack message
            return self._handle_generic_activity(eng_plan, item, date_str)

    # ─── ACTIVITY HANDLERS ───────────────────────────────────────────────────

    def _handle_ticket_progress(
        self,
        eng_plan: EngineerDayPlan,
        item: AgendaItem,
        date_str: str,
    ) -> List[str]:
        """Simulates an engineer working on a specific JIRA ticket and potentially opening a PR."""
        import json # Ensure json is available
        
        assignee = eng_plan.name
        ticket_id = item.related_id
        if not ticket_id:
            return []
        
        # Find the ticket in state
        ticket = next((t for t in self._state.jira_tickets if t["id"] == ticket_id), None)
        if not ticket:
            return [eng_plan.name]

        current_actor_time, new_cursor = self._clock.advance_actor(assignee, hours=2.0)
        current_actor_time_iso = current_actor_time.isoformat()
        ctx = self._mem.context_for_prompt(
            f"{ticket_id} {ticket['title']}", n=3, as_of_time=current_actor_time_iso
        )
        linked_prs = ticket.get("linked_prs", [])

        # Build structured ticket context — title + full state in one object
        if self._registry:
            ticket_ctx = self._registry.ticket_summary(
                ticket, self._state.day
            ).for_prompt()
        else:
            # Graceful fallback if registry not wired yet
            ticket_ctx = (
                f"Ticket: [{ticket_id}] {ticket.get('title', '')}\n"
                f"Status: {ticket.get('status', 'To Do')}\n"
                f"Recent comments: " + (
                    "\n".join(
                        f"  - {c['author']} ({c['date']}): {c['text']}"
                        for c in ticket.get("comments", [])[-3:]
                    ) or "None."
                )
            )

        backstory = self._persona_helper(assignee, mem=self._mem, graph_dynamics=self._gd)

        agent = Agent(
            role="Software Engineer",
            backstory=backstory,
            goal="Make progress on the ticket and report status.",
            llm=self._worker,
        )
        task = Task(
            description=(
                f"{ticket_ctx}\n\n"
                f"Your specific task today: {item.description}\n"
                f"Memory context: {ctx}\n\n"
                f"1. Write a 1-3 sentence JIRA comment about what you did today.\n"
                f"2. Decide if the coding phase is completely finished today.\n"
                f"   (If this is day 1 of a complex ticket, it's likely false.)\n"
                f"Respond ONLY with valid JSON:\n"
                f'{{\n'
                f'  "comment": "string",\n'
                f'  "is_code_complete": boolean\n'
                f'}}'
            ),
            expected_output="Valid JSON only.",
            agent=agent,
        )

        raw_result = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()
        clean_json = raw_result.replace("```json", "").replace("```", "").strip()
        
        try:
            parsed_data = json.loads(clean_json)
            comment_text = parsed_data.get("comment", f"Worked on {ticket_id}.")
            is_code_complete = parsed_data.get("is_code_complete", False)
        except json.JSONDecodeError:
            comment_text = clean_json # Fallback if LLM fails JSON
            is_code_complete = False

        BLOCKER_KEYWORDS = ("blocked", "blocker", "waiting on", "can't proceed", "stuck")
        if any(kw in comment_text.lower() for kw in BLOCKER_KEYWORDS):
            self._mem.log_event(SimEvent(
                type="blocker_flagged",
                timestamp=current_actor_time_iso,
                day=self._state.day,
                date=date_str,
                actors=[assignee],
                artifact_ids={"jira": ticket_id},
                facts={"ticket_id": ticket_id, "comment": comment_text},
                summary=f"{assignee} flagged a blocker on {ticket_id}.",
                tags=["jira", "blocker"]
            ))

        # 3. Update the JIRA ticket
        ticket.setdefault("comments", []).append({
            "author": assignee,
            "date": date_str,
            "created": current_actor_time_iso,
            "updated": current_actor_time_iso,
            "text": f"\"{comment_text}\"",
            "day": self._state.day
        })
        
        # Ensure status is at least In Progress
        if ticket["status"] == "To Do":
            ticket["status"] = "In Progress"

        # 4. Spawning the PR!
        spawned_pr_id = None
        if is_code_complete and not linked_prs:
            # Code is done, generate the PR!
            pr = self._git.create_pr(
                author=assignee, 
                ticket_id=ticket_id, 
                title=f"[{ticket_id}] {ticket['title'][:50]}", 
                timestamp=current_actor_time_iso
            )
            spawned_pr_id = pr["pr_id"]
            ticket.setdefault("linked_prs", []).append(spawned_pr_id)
            ticket["status"] = "In Review" # Advance the status

        ticket["updated_at"] = current_actor_time_iso

        # 5. Save and Embed
        self._save_ticket(ticket)
        self._mem.embed_artifact(
            id=f"{ticket_id}_comment_{len(ticket['comments'])}",
            type="jira_comment",
            title=f"Comment on {ticket_id}",
            content=comment_text,
            day=self._state.day, date=date_str,
            metadata={"ticket_id": ticket_id, "author": assignee},
            timestamp=current_actor_time_iso
        )

        artifacts = {
            "jira": ticket_id,
            "jira_comment": f"{ticket_id}_comment_{len(ticket['comments'])}"
        }
        if spawned_pr_id:
            artifacts["pr"] = spawned_pr_id

        self._mem.log_event(SimEvent(
            type="ticket_progress",
            timestamp=current_actor_time_iso,
            day=self._state.day, 
            date=date_str,
            actors=[assignee],
            artifact_ids=artifacts,
            facts={"ticket_id": ticket_id, "status": ticket["status"], "spawned_pr": spawned_pr_id},
            summary=f"{assignee} worked on {ticket_id}. " + (f"Opened PR {spawned_pr_id}!" if spawned_pr_id else ""),
            tags=["jira", "engineering"]
        ))

        bucket = self._state.ticket_actors_today.setdefault(ticket_id, set())
        bucket.add(assignee)

        generated_artifacts = [ticket_id]
        if spawned_pr_id is not None:
            generated_artifacts.append(spawned_pr_id)
            
        return generated_artifacts

    def _handle_pr_review(
        self,
        eng_plan: EngineerDayPlan,
        item:     AgendaItem,
        date_str: str,
    ) -> List[str]:
        """
        Engineer reviews a PR.
        Generates: GitHub review comment thread in Slack #engineering.
        """
        reviewer = eng_plan.name
        pr_id    = item.related_id

        # Find the PR — if no specific ID, pick an open one the reviewer is on
        pr = self._find_pr(pr_id) or self._find_reviewable_pr(reviewer)
        if not pr:
            return [reviewer]

        author   = pr.get("author", reviewer)
        pr_title = pr.get("title", "Unknown PR")
        artifact_time, new_cursor = self._clock.advance_actor(author, hours=item.estimated_hrs)
        current_actor_time = artifact_time.isoformat()

        ctx      = self._mem.context_for_prompt(pr_title, n=2, as_of_time=current_actor_time)

        # Generate review comment
        agent = Agent(
            role="Code Reviewer",
            goal="Write a realistic PR review comment.",
            backstory=(
                f"You are {reviewer}, reviewing {author}'s PR: {pr_title}. "
                f"{self._gd.stress_tone_hint(reviewer)}"
            ),
            llm=self._worker,
        )
        task = Task(
            description=(
                f"Write a GitHub PR review comment for: {pr_title}\n"
                f"Context: {ctx}\n"
                f"1-4 sentences. Be specific — mention code patterns, potential "
                f"edge cases, or ask a clarifying question. "
                f"Tone reflects the reviewer's current stress level."
            ),
            expected_output="A realistic code review comment.",
            agent=agent,
        )
        review_text = str(
            Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        ).strip()

        import os, json as _json
        pr_path = f"{self._base}/git/prs/{pr.get('pr_id', pr_id)}.json"
        os.makedirs(os.path.dirname(pr_path), exist_ok=True)
        with open(pr_path, "w") as f:
            _json.dump(pr, f, indent=2)

        pr_comment = {
            "author": reviewer,
            "date": date_str,
            "timestamp": current_actor_time,
            "text": review_text
        }
        pr.setdefault("comments", []).append(pr_comment)



        # Emit as a GitHub bot message in #engineering
        self._emit_bot_message(
            "engineering",
            "GitHub",
            f"💬 {reviewer} reviewed {pr.get('pr_id', pr_id)}: \"{review_text[:120]}\"",
            current_actor_time
        )

        # Author acknowledges in Slack if the review raises a question
        actors = [reviewer, author]
        if "?" in review_text:
            actors = self._emit_review_reply(
                author, reviewer, pr.get("pr_id", "PR"), review_text, date_str,
                current_actor_time
            )

        # Boost PR review edge
        self._gd.record_pr_review(author, [reviewer])

        self._mem.log_event(SimEvent(
            type="pr_review",
            timestamp=current_actor_time,
            day=self._state.day,
            date=date_str,
            actors=actors,
            artifact_ids={"pr": pr.get("pr_id", pr_id or "")},
            facts={
                "reviewer":    reviewer,
                "author":      author,
                "pr_title":    pr_title,
                "review_text": review_text,
                "has_question": "?" in review_text,
            },
            summary=f"{reviewer} reviewed {pr.get('pr_id', 'PR')} by {author}.",
            tags=["pr_review", "engineering"],
        ))

        logger.info(f"    [dim]🔍 {reviewer} reviewed {pr.get('pr_id', 'PR')}[/dim]")
        return actors

    def _handle_one_on_one(
        self,
        eng_plan: EngineerDayPlan,
        item:     AgendaItem,
        date_str: str,
    ) -> List[str]:
        """
        Engineer has a 1:1 with their lead or a collaborator.
        Generates: DM thread (2-4 messages).
        """
        name         = eng_plan.name
        collaborator = next(iter(item.collaborator), None) or self._find_lead_for(name)
        participants = [name, collaborator]
        if not collaborator or collaborator == name:
            return [name]
        
        meeting_start, meeting_end = self._clock.sync_and_advance(
            participants, 
            hours=item.estimated_hrs
        )
        meeting_time_iso = meeting_start.isoformat()

        stress_a = self._gd.stress_tone_hint(name)
        stress_b = self._gd.stress_tone_hint(collaborator)
        ctx      = self._mem.context_for_prompt(
            f"1on1 {name} {collaborator} workload", n=2, as_of_time=meeting_time_iso
        )

        agent = Agent(
            role="Workplace Conversation Writer",
            goal="Write a realistic 1:1 DM exchange.",
            backstory="You write authentic, unscripted workplace conversations.",
            llm=self._worker,
        )
        task = Task(
            description=(
                f"Write a 3-5 message DM between {name} and {collaborator}.\n"
                f"{name}: {stress_a}\n"
                f"{collaborator}: {stress_b}\n"
                f"Context: {ctx}\n"
                f"Topics might include: workload, current sprint, a decision that "
                f"needs to be made, or something personal-professional (lunch, PTO).\n"
                f"Format EXACTLY: Name: [Message]"
            ),
            expected_output="DM transcript. Name: [Message] format.",
            agent=agent,
        )
        result = str(
            Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        )

        messages = self._parse_slack_messages(
            result, [name, collaborator], hour_range=(9, 11)
        )
        if not messages:
            return [name, collaborator]
        

        current_msg_time = datetime.fromisoformat(meeting_time_iso)

        for msg in messages:
            # Stamp the message with our exact, mathematically safe time
            msg["ts"] = current_msg_time.isoformat()
            
            # Add 1 to 4 random minutes so the reply looks like a human typing
            current_msg_time += timedelta(minutes=random.randint(1, 4))

        p1, p2    = sorted([name, collaborator])
        channel   = f"dm_{p1.lower()}_{p2.lower()}"
        path      = f"{self._base}/slack/channels/{channel}/{date_str}_1on1.json"
        self._save_slack(path, messages, channel)

        self._mem.log_event(SimEvent(
            type="1on1",
            timestamp=meeting_time_iso,
            day=self._state.day,
            date=date_str,
            actors=[name, collaborator],
            artifact_ids={"slack": path},
            facts={"participants": [name, collaborator], "message_count": len(messages)},
            summary=f"1:1 between {name} and {collaborator}.",
            tags=["1on1", "slack"],
        ))

        self._gd.record_slack_interaction([name, collaborator])
        logger.info(f"    [dim]👥 1:1 {name} ↔ {collaborator}[/dim]")
        return [name, collaborator]

    def _handle_async_question(
        self,
        eng_plan:  EngineerDayPlan,
        item:      AgendaItem,
        dept_plan: DepartmentDayPlan,
        date_str:  str,
    ) -> List[str]:
        """
        Engineer asks a question in a channel.
        Generates: Slack thread with 2-4 replies from colleagues.
        The question is grounded in the engineer's current ticket or blocker.
        """
        asker        = eng_plan.name
        collaborator = next(iter(item.collaborator), None) or self._closest_colleague(asker)
        ticket_id    = item.related_id
        ticket       = self._find_ticket(ticket_id)
        ticket_title = ticket["title"] if ticket else item.description

        # Pick the channel — same dept = dept channel, cross-dept = digital-hq
        initial_participants = [asker]
        if collaborator:
            initial_participants.append(collaborator)
            
        depts = {dept_of_name(p, self._org_chart) for p in initial_participants}
        
        if len(depts) > 1:
            channel = "digital-hq"
        else:
            channel = dept_of_name(asker, self._org_chart).lower().replace(" ", "-")

        # Pull in 1-2 more people from the channel naturally
        channel_members = self._channel_members(channel, asker)
        responders = [collaborator] if collaborator else []
        extras = [n for n in channel_members if n != asker and n not in responders]
        responders.extend(random.sample(extras, min(1, len(extras))))
        all_actors = list({asker} | set(responders))

        chat_duration_mins = random.randint(5, 20)
        chat_duration_hours = chat_duration_mins / 60.0

        meeting_start, meeting_end = self._clock.sync_and_advance(
            all_actors, 
            hours=chat_duration_hours
        )
        meeting_time_iso = meeting_start.isoformat()

        ctx = self._mem.context_for_prompt(ticket_title, n=2, as_of_time=meeting_time_iso)

        agent = Agent(
            role="Slack Thread Writer",
            goal="Write a realistic async question thread.",
            backstory="You write authentic Slack conversations in engineering teams.",
            llm=self._worker,
        )
        profiles = "\n".join(
            f"  - {n} ({dept_of_name(n, self._org_chart)}): "
            f"{self._gd.stress_tone_hint(n)}"
            for n in all_actors
        )
        task = Task(
            description=(
                f"{asker} has a question related to: {ticket_title}\n"
                f"Participants:\n{profiles}\n"
                f"Context: {ctx}\n\n"
                f"Write a 3-5 message Slack thread. {asker} asks first, "
                f"others respond with genuine help, follow-up questions, or "
                f"'I don't know but try X'. Keep it realistic — not everyone "
                f"knows the answer immediately.\n"
                f"Format EXACTLY: Name: [Message]"
            ),
            expected_output="Slack thread. Name: [Message] format.",
            agent=agent,
        )
        result = str(
            Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        )

        messages = self._parse_slack_messages(result, all_actors, hour_range=(10, 16))
        if not messages:
            return all_actors

        current_msg_time = datetime.fromisoformat(meeting_time_iso)

        for msg in messages:
            # Stamp the message with our exact, mathematically safe time
            msg["ts"] = current_msg_time.isoformat()
            
            # Add 1 to 4 random minutes so the reply looks like a human typing
            current_msg_time += timedelta(minutes=random.randint(1, 4))

        path = f"{self._base}/slack/channels/{channel}/{date_str}_{asker.lower()}_q.json"
        self._save_slack(path, messages, channel)

        self._mem.log_event(SimEvent(
            type="async_question",
            timestamp=meeting_time_iso,
            day=self._state.day,
            date=date_str,
            actors=all_actors,
            artifact_ids={"slack": path, "jira": ticket_id or ""},
            facts={
                "asker":     asker,
                "channel":   channel,
                "topic":     ticket_title,
                "responders": responders,
                "message_count": len(messages),
            },
            summary=f"{asker} asked a question in #{channel} about {ticket_title[:50]}.",
            tags=["async_question", "slack"],
        ))

        self._gd.record_slack_interaction(all_actors)
        logger.info(f"    [dim]❓ {asker} → #{channel} ({len(messages)} msgs)[/dim]")
        return all_actors

    def _handle_design_discussion(
        self,
        eng_plan:  EngineerDayPlan,
        item:      AgendaItem,
        dept_plan: DepartmentDayPlan,
        date_str:  str,
    ) -> List[str]:
        """
        Small group design discussion — typically 2-3 engineers.
        Generates: Slack thread + optional Confluence stub.
        """
        initiator = eng_plan.name
        collaborators = item.collaborator or (
            [c] if (c := self._closest_colleague(initiator)) else []
        )
        participants = list({initiator} | set(collaborators))

        # Pull one more person with relevant expertise if available
        if len(participants) < 3:
            topic_lower = item.description.lower()
            for name in self._all_names:
                if name in participants:
                    continue
                p = self._config.get("personas", {}).get(name, {})
                expertise = [e.lower() for e in p.get("expertise", [])]
                if any(kw in topic_lower for kw in expertise):
                    participants.append(name)
                    break

        chat_duration_mins = random.randint(5, 45)
        chat_duration_hours = chat_duration_mins / 60.0

        meeting_start, meeting_end = self._clock.sync_and_advance(
            participants, 
            hours=chat_duration_hours
        )
        meeting_time_iso = meeting_start.isoformat()

        ctx = self._mem.context_for_prompt(item.description, n=3, as_of_time=meeting_time_iso)

        agent = Agent(
            role="Engineering Team",
            goal="Write a realistic design discussion Slack thread.",
            backstory="You write authentic technical discussions between engineers.",
            llm=self._planner,
        )
        profiles = "\n".join(
            f"  - {n}: {self._gd.stress_tone_hint(n)}" for n in participants
        )
        task = Task(
            description=(
                f"Write a 5-8 message Slack design discussion about: {item.description}\n"
                f"Participants:\n{profiles}\n"
                f"Context: {ctx}\n\n"
                f"This should feel like engineers working through a real technical "
                f"decision — trade-offs, constraints, 'what about X' questions. "
                f"Someone should land on a tentative conclusion or next step.\n"
                f"Format EXACTLY: Name: [Message]"
            ),
            expected_output="Slack thread. Name: [Message] format.",
            agent=agent,
        )
        result = str(
            Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        )

        messages = self._parse_slack_messages(result, participants, hour_range=(10, 15))

        depts = {dept_of_name(p, self._org_chart) for p in participants}
        if len(depts) > 1:
            dept_channel = "digital-hq"
        else:
            dept_channel = dept_of_name(initiator, self._org_chart).lower().replace(" ", "-")

        path = (
            f"{self._base}/slack/channels/{dept_channel}/"
            f"{date_str}_{initiator.lower()}_design.json"
        )

        if messages:
            current_msg_time = datetime.fromisoformat(meeting_time_iso)

            for msg in messages:
                # Stamp the message with our exact, mathematically safe time
                msg["ts"] = current_msg_time.isoformat()
                
                # Add 1 to 4 random minutes so the reply looks like a human typing
                current_msg_time += timedelta(minutes=random.randint(1, 8))

            self._save_slack(path, messages, dept_channel)

        # 30% chance a design discussion spawns a Confluence stub
        conf_id = None
        if random.random() < 0.30 and messages:
            # pass the generated Slack messages into the doc generator
            conf_id = self._create_design_doc_stub(
                initiator, participants, item.description, ctx, date_str, messages
            )

        self._mem.log_event(SimEvent(
            type="design_discussion",
            timestamp=meeting_time_iso,
            day=self._state.day,
            date=date_str,
            actors=participants,
            artifact_ids={"slack": path, "confluence": conf_id or ""},
            facts={
                "topic":         item.description,
                "participants":  participants,
                "spawned_doc":   conf_id is not None,
                "message_count": len(messages),
            },
            summary=(
                f"{initiator} led design discussion on '{item.description[:50]}' "
                f"with {', '.join(p for p in participants if p != initiator)}."
            ),
            tags=["design_discussion", "slack"],
        ))

        self._gd.record_slack_interaction(participants)
        logger.info(
            f"    [dim]🏗️  Design discussion: {item.description[:40]} "
            f"({len(participants)} engineers)[/dim]"
        )
        return participants

    def _handle_mentoring(
        self,
        eng_plan: EngineerDayPlan,
        item:     AgendaItem,
        date_str: str,
    ) -> List[str]:
        """
        Senior engineer mentors a junior colleague.
        Generates: DM thread. Boosts social graph edge significantly.
        """
        mentor   = eng_plan.name
        mentee = next(iter(item.collaborator), None) or self._find_junior_colleague(mentor)
        if not mentee or mentee == mentor:
            return [mentor]
        
        participants = [mentor, mentee]

        session_mins = random.randint(30, 90)
        session_hours = session_mins / 60.0
        meeting_start, meeting_end = self._clock.sync_and_advance(
            participants, 
            hours=session_hours
        )
        meeting_time_iso = meeting_start.isoformat()

        ctx = self._mem.context_for_prompt(
            f"mentoring {mentee} learning growth", n=2, as_of_time=meeting_time_iso
        )

        agent = Agent(
            role="Workplace Mentor",
            goal="Write a realistic mentoring DM exchange.",
            backstory=(
                f"You are {mentor}, an experienced engineer helping {mentee} grow. "
                f"{self._gd.stress_tone_hint(mentor)}"
            ),
            llm=self._worker,
        )
        task = Task(
            description=(
                f"Write a 4-6 message DM between mentor {mentor} and mentee {mentee}.\n"
                f"Context: {ctx}\n"
                f"Topics: career growth, a technical concept {mentee} is learning, "
                f"feedback on recent work, or advice on handling a situation.\n"
                f"Format EXACTLY: Name: [Message]"
            ),
            expected_output="DM transcript. Name: [Message] format.",
            agent=agent,
        )
        result = str(
            Crew(agents=[agent], tasks=[task], verbose=False).kickoff()
        )

        messages = self._parse_slack_messages(
            result, [mentor, mentee], hour_range=(14, 17)
        )
        if not messages:
            return [mentor, mentee]

        p1, p2  = sorted([mentor, mentee])
        channel = f"dm_{p1.lower()}_{p2.lower()}"
        path    = f"{self._base}/slack/channels/{channel}/{date_str}_mentoring.json"
        self._save_slack(path, messages, channel)

        # Mentoring is a strong relationship signal
        self._gd.record_slack_interaction([mentor, mentee])
        self._gd.record_slack_interaction([mentor, mentee])  # double boost

        self._mem.log_event(SimEvent(
            type="mentoring",
            timestamp=meeting_time_iso,
            day=self._state.day,
            date=date_str,
            actors=[mentor, mentee],
            artifact_ids={"slack": path},
            facts={"mentor": mentor, "mentee": mentee, "message_count": len(messages)},
            summary=f"{mentor} mentored {mentee}.",
            tags=["mentoring", "slack"],
        ))

        logger.info(f"    [dim]🎓 {mentor} → {mentee} (mentoring)[/dim]")
        return [mentor, mentee]

    def _handle_generic_activity(
        self,
        eng_plan: EngineerDayPlan,
        item:     AgendaItem,
        date_str: str,
    ) -> List[str]:
        """
        Fallback for unknown activity types — generates a short Slack mention.
        """
        name    = eng_plan.name
        channel = dept_of_name(name, self._org_chart).lower().replace(" ", "-")

        cron_time_iso = self._clock.now("system").isoformat()

        self._emit_bot_message(
            channel,
            name,
            f"Working on: {item.description}",
            cron_time_iso
        )
        return [name]
    
    def _handle_collision_event(self, event: ProposedEvent, date_str: str):
        """Renders the unplanned interaction as a Slack thread."""
        participants = event.actors
        
        # Determine the vibe from the coordinator's facts_hint
        tension = event.facts_hint.get("tension_level", "medium")
        
        # Build profiles with live stress and persona backstories
        profiles = []
        for name in participants:
            backstory = self._persona_helper(name, mem=self._mem, graph_dynamics=self._gd)
            profiles.append(f"NAME: {name}\nCONTEXT: {backstory}")

        # Use the Planner model for complex group dynamics
        agent = Agent(
            role="Org Interaction Simulator",
            goal=f"Simulate a {tension}-tension interaction: {event.event_type}",
            backstory="You write authentic corporate dialogue based on persona stress.",
            llm=self._planner
        )

        task = Task(
            description=(
                f"TOPIC: {event.rationale}\n"
                f"TENSION LEVEL: {tension}\n"
                f"PARTICIPANTS:\n{chr(10).join(profiles)}\n\n"
                f"Write a 5-8 message Slack thread. If tension is 'low', focus on "
                f"synergy and mentorship. If 'high', focus on friction and blockers. "
                f"STRICTLY follow each person's TYPING QUIRKS (caps, punctuation, etc)."
            ),
            expected_output="Slack transcript. Name: [Message] format.",
            agent=agent
        )

        result = str(Crew(agents=[agent], tasks=[task]).kickoff())
        messages = self._parse_slack_messages(result, participants)
        
        # Save to digital-hq or relevant channel
        channel = "digital-hq"
        path = f"{self._base}/slack/channels/{channel}/{date_str}_collision.json"
        self._save_slack(path, messages, channel)
        
        # Log the simulation event for the memory store
        self._mem.log_event(SimEvent(
            type="org_collision",
            timestamp=self._clock.now("system").isoformat(),
            day=self._state.day, date=date_str,
            actors=participants,
            artifact_ids={"slack": path},
            facts={"tension": tension, "type": event.event_type},
            summary=f"Unplanned {tension} interaction: {event.rationale}",
            tags=["collision", tension]
        ))

    # ─── ARTIFACT GENERATORS ─────────────────────────────────────────────────

    def _emit_blocker_slack(
        self,
        asker:       str,
        collaborator: str,
        ticket_id:   str,
        ticket_title: str,
        blocker_text: str,
        date_str:    str,
        timestamp: str
    ) -> List[str]:
        """Short 2-message Slack exchange when an engineer is blocked."""
        asker_dept = dept_of_name(asker, self._org_chart)
        channel    = asker_dept.lower().replace(" ", "-")

        agent = Agent(
            role="Engineer",
            goal="Write a realistic blocker Slack message.",
            backstory=f"You are {asker}. {self._gd.stress_tone_hint(asker)}",
            llm=self._worker,
        )
        task = Task(
            description=(
                f"Write 2 Slack messages: {asker} mentions being blocked on "
                f"[{ticket_id}]: {ticket_title}, then {collaborator} responds.\n"
                f"Blocker context: {blocker_text[:100]}\n"
                f"Format: Name: [Message]"
            ),
            expected_output="Two Slack messages.",
            agent=agent,
        )
        result  = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())
        messages = self._parse_slack_messages(
            result, [asker, collaborator], hour_range=(10, 15)
        )

        if messages:
            

            current_msg_time = datetime.fromisoformat(timestamp)

            for msg in messages:
                # Stamp the message with our exact, mathematically safe time
                msg["ts"] = current_msg_time.isoformat()
                
                # Add 1 to 4 random minutes so the reply looks like a human typing
                current_msg_time += timedelta(minutes=random.randint(1, 4))

            path = (
                f"{self._base}/slack/channels/{channel}/"
                f"{date_str}_{asker.lower()}_blocked.json"
            )
            self._save_slack(path, messages, channel)
            self._gd.record_slack_interaction([asker, collaborator])

            self._mem.log_event(SimEvent(
                type="blocker_flagged",
                day=self._state.day,
                date=date_str,
                timestamp=timestamp,
                actors=[asker, collaborator],
                artifact_ids={"jira": ticket_id},
                facts={"blocker_reason": blocker_text},
                summary=f"{asker} is blocked on {ticket_id}, pinged {collaborator}.",
                tags=["slack", "blocker"]
            ))

        return [asker, collaborator]

    def _emit_review_reply(
        self,
        author:    str,
        reviewer:  str,
        pr_id:     str,
        review_text: str,
        date_str:  str,
        timestamp:   str
    ) -> List[str]:
        """Author replies to a review question in #engineering."""
        agent = Agent(
            role="PR Author",
            goal="Reply to a code review question.",
            backstory=f"You are {author}. {self._gd.stress_tone_hint(author)}",
            llm=self._worker,
        )
        task = Task(
            description=(
                f"{reviewer} asked in a PR review: {review_text[:120]}\n"
                f"Write {author}'s reply — answer the question, clarify intent, "
                f"or push back if you disagree. 1-2 sentences.\n"
                f"Format: {author}: [reply]"
            ),
            expected_output="One reply message.",
            agent=agent,
        )
        reply = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff()).strip()

        self._emit_bot_message(
            "engineering", "GitHub",
            f"💬 {author} replied to {reviewer}'s review on {pr_id}: "
            f"\"{reply[:100]}\"",
            timestamp
        )
        return [author, reviewer]

    def _create_design_doc_stub(
        self,
        author:           str,
        participants:     List[str],
        topic:            str,
        ctx:              str,              # kept for signature compat, not used
        date_str:         str,
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

    # ─── LOGGING HELPERS ─────────────────────────────────────────────────────

    def _log_deferred_item(
        self, name: str, item: AgendaItem, date_str: str
    ) -> None:
        """Log a deferred agenda item so the record shows the interruption."""
        current_time_iso = self._clock.now(name).isoformat()

        self._mem.log_event(SimEvent(
            type="agenda_item_deferred",
            timestamp=current_time_iso,
            day=self._state.day,
            date=date_str,
            actors=[name],
            artifact_ids={"jira": item.related_id or ""},
            facts={
                "name":          name,
                "activity_type": item.activity_type,
                "description":   item.description,
                "defer_reason":  item.defer_reason or "unspecified",
            },
            summary=(
                f"{name}'s '{item.description[:50]}' deferred: "
                f"{item.defer_reason or 'unspecified'}"
            ),
            tags=["deferred", "agenda"],
        ))

    def _log_deep_work(
        self, name: str, item: AgendaItem, date_str: str
    ) -> None:
        
        # 1. Deep work takes time! Advance their cursor by the estimated hours.
        # This prevents anyone else from scheduling a 1-on-1 with them during this block.
        artifact_time, new_cursor = self._clock.advance_actor(name, hours=item.estimated_hrs)

        self._mem.log_event(SimEvent(
            type="deep_work_session",
            timestamp=artifact_time.isoformat(),
            day=self._state.day,
            date=date_str,
            actors=[name],
            artifact_ids={},
            facts={"name": name, "focus": item.description},
            summary=f"{name} in deep work: {item.description[:50]}",
            tags=["deep_work"],
        ))

    # ─── AMBIENT SIGNALS (unchanged from original) ────────────────────────────

    def _maybe_bot_alerts(self) -> None:
        cron_time_iso = self._clock.now("system").isoformat()

        if random.random() < self._config["simulation"].get("aws_alert_prob", 0.4):
            legacy = self._config.get("legacy_system", {})
            self._emit_bot_message(
                "system-alerts", "AWS Cost Explorer",
                f"⚠️ Daily budget threshold exceeded. "
                f"{legacy.get('aws_alert_message', 'Cloud costs remain elevated.')}",
                cron_time_iso
            )
        elif random.random() < self._config["simulation"].get("snyk_alert_prob", 0.2):
            self._emit_bot_message(
                "engineering", "Snyk Security",
                "🔒 3 new medium-severity vulnerabilities detected in npm dependencies.",
                cron_time_iso
            )

    def _maybe_adhoc_confluence(self) -> None:
        if random.random() >= self._config["simulation"].get("adhoc_confluence_prob", 0.3):
            return
        if self._confluence is None:
            return
        author    = random.choice(self._all_names)
        backstory = self._persona_helper(author, mem=self._mem, graph_dynamics=self._gd)
        self._confluence.write_adhoc_page(author=author, backstory=backstory)

    def _trigger_watercooler_chat(self, target_actor: str, date_str: str) -> None:
        """Injects non-work chatter, pulling the target actor away from their work."""
        if target_actor not in self._graph:
            return
            
        edges = self._graph[target_actor]
        if not edges:
            return
            
        # Pull 1-2 work friends to distract them
        colleagues = random.choices(
            list(edges.keys()),
            weights=[edges[n]["weight"] for n in edges.keys()],
            k=random.randint(1, 2),
        )
        participants = list(set([target_actor] + colleagues))
        if len(participants) < 2:
            return

        # 1. The Distraction Time Sink
        # This uses sim_clock to find the busiest person in this group, syncs 
        # them together, and eats up 10-15 minutes of their day.
        chat_duration_mins = random.randint(10, 15)
        chat_duration_hours = chat_duration_mins / 60.0
        thread_start, thread_end = self._clock.sync_and_advance(participants, hours=chat_duration_hours)
        thread_start_iso = thread_start.isoformat()

        # 2. LLM Prompting
        topics = [
            "weekend plans", "a trending TV show", "complaining about the weather", 
            "trying to figure out lunch options", "sharing a pet photo"
        ]
        topic = random.choice(topics)
        
        profiles = [f"{n} ({dept_of_name(n, self._org_chart)}): {self._gd.stress_tone_hint(n)}" for n in participants]

        agent = Agent(
            role="Slack Observer",
            goal="Write a casual, non-work Slack thread.",
            backstory="You observe real humans taking a break from work.",
            llm=self._worker,
        )
        task = Task(
            description=(
                f"Write a 3-5 message Slack conversation between:\n"
                f"{chr(10).join(profiles)}\n\n"
                f"Topic: {topic}\n"
                f"This MUST be entirely unrelated to work, code, or Jira tickets. "
                f"Format EXACTLY: Name: [Message]"
            ),
            expected_output="Slack conversation.",
            agent=agent,
        )
        
        result = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())
        messages = self._parse_slack_messages(result, participants)

        if messages:
            current_msg_time = datetime.fromisoformat(thread_start_iso)
            for msg in messages:
                msg["ts"] = current_msg_time.isoformat()
                current_msg_time += timedelta(minutes=random.randint(1, 3))

            channel = "random" if len(participants) > 2 else f"dm_{sorted(participants)[0].lower()}_{sorted(participants)[1].lower()}"
            path = f"{self._base}/slack/channels/{channel}/{date_str}_{target_actor}_distracted.json"
            
            self._save_slack(path, messages, channel)
            self._gd.record_slack_interaction(participants)
            
            self._mem.log_event(SimEvent(
                type="watercooler_chat",
                timestamp=thread_start_iso,
                day=self._state.day,
                date=date_str,
                actors=participants,
                artifact_ids={"slack": path},
                facts={"topic": topic, "message_count": len(messages)},
                summary=f"{target_actor} got distracted chatting about {topic} with {len(participants)-1} others.",
                tags=["watercooler", "slack", "distraction"],
            ))
            
            logger.info(f"    [dim]☕ Distraction: {target_actor} pulled into chat about {topic}[/dim]")

    def _legacy_slack_chatter(self, date_str: str) -> None:
        """Original _handle_normal_day() behaviour — used as fallback."""
        seed_person = random.choice(self._all_names)
        edges       = self._graph[seed_person]
        colleagues  = random.choices(
            list(edges.keys()),
            weights=[edges[n]["weight"] for n in edges.keys()],
            k=random.randint(2, 4),
        )
        participants = list(set([seed_person] + colleagues))
        # 1. Randomize the duration between 5 and 30 minutes
        #    (Convert to hours because sync_and_advance expects hours)
        chat_duration_mins = random.randint(5, 30)
        chat_duration_hours = chat_duration_mins / 60.0

        thread_start, thread_end = self._clock.sync_and_advance(participants, hours=chat_duration_hours)
        thread_start_iso = thread_start.isoformat()

        # 2. Mathematically restrict the context to the exact moment the chat begins
        ctx = self._mem.context_for_prompt(
            self._state.daily_theme, 
            n=2, 
            as_of_time=thread_start_iso 
        )

        profiles = [f"{n} ({dept_of_name(n, self._org_chart)})" for n in participants]

        agent = Agent(
            role="Slack Observer",
            goal="Write casual Slack thread.",
            backstory="You observe real humans chatting.",
            llm=self._worker,
        )
        task = Task(
            description=(
                f"Write a 4-6 message Slack conversation between:\n"
                f"{chr(10).join(profiles)}\n\n"
                f"Theme: {self._state.daily_theme}\nContext: {ctx}\n"
                f"Format EXACTLY: Name: [Message]"
            ),
            expected_output="Slack conversation.",
            agent=agent,
        )
        result   = str(Crew(agents=[agent], tasks=[task], verbose=False).kickoff())
        messages = self._parse_slack_messages(result, participants)

        if messages:
            depts    = {dept_of_name(p, self._org_chart) for p in participants}
            channel  = (
                f"dm_{sorted(participants)[0].lower()}_{sorted(participants)[1].lower()}"
                if len(participants) == 2
                else (list(depts)[0].lower().replace(" ", "-") if len(depts) == 1
                      else "digital-hq")
            )
            path = f"{self._base}/slack/channels/{channel}/{date_str}_{seed_person}.json"
            self._save_slack(path, messages, channel)
            self._gd.record_slack_interaction(participants)

    # ─── LOW-LEVEL UTILITIES ──────────────────────────────────────────────────

    def _parse_slack_messages(
        self,
        raw:         str,
        valid_names: List[str],
        hour_range:  Tuple[int, int] = (9, 17),
    ) -> List[dict]:
        messages = []
        for line in raw.split("\n"):
            if ":" not in line:
                continue
            name, text = line.split(":", 1)
            name = name.strip()
            if name in valid_names:
                messages.append({
                    "user": name,
                    "text": text.strip(),
                    "ts":   self._state.current_date.replace(
                        hour=random.randint(*hour_range),
                        minute=random.randint(0, 59),
                    ).isoformat(),
                })
        return messages

    def _save_slack(self, path: str, messages: List[dict], channel: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(messages, f, indent=2)
        self._state.slack_threads.append({
            "date":          str(self._state.current_date.date()),
            "channel":       channel,
            "message_count": len(messages),
        })

    def _save_md(self, path: str, content: str) -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    def _save_ticket(self, ticket: dict) -> None:
        import os, json as _json
        path = f"{self._base}/jira/{ticket['id']}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            _json.dump(ticket, f, indent=2)

    def _emit_bot_message(self, channel: str, bot_name: str, text: str, timestamp: str) -> None:
        import os
        date_str   = str(self._state.current_date.date())
        slack_path = f"{self._base}/slack/channels/{channel}/{date_str}_bots.json"
        messages   = []
        if os.path.exists(slack_path):
            with open(slack_path) as f:
                messages = json.load(f)
        messages.append({
            "user":   bot_name,
            "email":  f"{bot_name.lower()}@bot.{self._domain}",
            "text":   text,
            "ts":     self._state.current_date.replace(
                hour=random.randint(8, 17),
                minute=random.randint(0, 59),
            ).isoformat(),
            "is_bot": True,
        })
        os.makedirs(os.path.dirname(slack_path), exist_ok=True)
        with open(slack_path, "w") as f:
            json.dump(messages, f, indent=2)

    def _find_ticket(self, ticket_id: Optional[str]) -> Optional[dict]:
        if not ticket_id:
            return None
        return next(
            (t for t in self._state.jira_tickets if t["id"] == ticket_id), None
        )

    def _find_pr(self, pr_id: Optional[str]) -> Optional[dict]:
        if not pr_id:
            return None
        import os
        path = f"{self._base}/git/prs/{pr_id}.json"
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def _find_reviewable_pr(self, reviewer: str) -> Optional[dict]:
        """Find an open PR where this person is listed as a reviewer."""
        import os, glob
        for path in glob.glob(f"{self._base}/git/prs/PR-*.json"):
            with open(path) as f:
                pr = json.load(f)
            if pr.get("status") == "open" and reviewer in pr.get("reviewers", []):
                return pr
        return None

    def _closest_colleague(self, name: str) -> Optional[str]:
        """Returns the highest-weight neighbour in the social graph."""
        if name not in self._graph:
            return None
        neighbours = [
            (n, self._graph[name][n].get("weight", 0))
            for n in self._graph.neighbors(name)
            if n in self._all_names
        ]
        if not neighbours:
            return None
        return max(neighbours, key=lambda x: x[1])[0]

    def _find_lead_for(self, name: str) -> Optional[str]:
        dept  = dept_of_name(name, self._org_chart)
        leads = self._config.get("leads", {})
        return leads.get(dept)

    def _find_junior_colleague(self, senior: str) -> Optional[str]:
        """Find a colleague with lower tenure — crude proxy for junior status."""
        dept    = dept_of_name(senior, self._org_chart)
        members = self._org_chart.get(dept, [])
        personas = self._config.get("personas", {})
        senior_tenure = personas.get(senior, {}).get("tenure", "mid")

        # Tenure ordering: intern < junior < mid < senior < staff < principal
        _RANK = {"intern": 0, "junior": 1, "mid": 2, "senior": 3,
                 "staff": 4, "principal": 5}
        senior_rank = _RANK.get(str(senior_tenure).lower().split()[0], 2)

        juniors = [
            n for n in members if n != senior
            and _RANK.get(
                str(personas.get(n, {}).get("tenure", "mid")).lower().split()[0], 2
            ) < senior_rank
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


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — dept lookup without importing flow globals
# ─────────────────────────────────────────────────────────────────────────────

def dept_of_name(name: str, org_chart: Dict[str, List[str]]) -> str:
    for dept, members in org_chart.items():
        if name in members:
            return dept
    return "Unknown"
