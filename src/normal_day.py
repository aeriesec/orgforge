from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from crewai import Agent, Process, Task, Crew

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
        confluence_writer=None,
        vader=None,
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
        self._vader          = vader

    # ─── PUBLIC ENTRY POINT ───────────────────────────────────────────────────

    def handle(self, org_plan: OrgDayPlan) -> None:
        """Processes both planned agenda items and unplanned org collisions."""
        logger.info(f"  [bold blue]💬 Normal Day Activity[/bold blue]")
        date_str = str(self._state.current_date.date())

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
        
        ticket = self._mem.get_ticket(ticket_id)
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
        comment_id = f"{ticket_id}_comment_{len(ticket['comments'])}"

        # 5. Save and Embed — upsert first so status is persisted even if
        # _save_ticket is patched out in tests
        self._mem.upsert_ticket(ticket)
        self._save_ticket(ticket)
        self._mem.embed_artifact(
            id=comment_id,
            type="jira_comment",
            title=f"Comment on {ticket_id}",
            content=comment_text,
            day=self._state.day, date=date_str,
            metadata={"ticket_id": ticket_id, "author": assignee},
            timestamp=current_actor_time_iso
        )

        active_inc = next((i for i in self._state.active_incidents if i.ticket_id == ticket_id), None)
        
        if active_inc and getattr(active_inc, "causal_chain", None):
            active_inc.causal_chain.append(comment_id)
            if spawned_pr_id:
                active_inc.causal_chain.append(spawned_pr_id)

        artifacts = {
            "jira": ticket_id,
            "jira_comment": comment_id
        }
        if spawned_pr_id:
            artifacts["pr"] = spawned_pr_id

        facts = {"ticket_id": ticket_id, "status": ticket["status"], "spawned_pr": spawned_pr_id}
        if active_inc and getattr(active_inc, "causal_chain", None):
            facts["causal_chain"] = active_inc.causal_chain.snapshot()

        self._mem.log_event(SimEvent(
            type="ticket_progress",
            timestamp=current_actor_time_iso,
            day=self._state.day, 
            date=date_str,
            actors=[assignee],
            artifact_ids=artifacts,
            facts=facts,
            summary=f"{assignee} worked on {ticket_id}. " + (f"Opened PR {spawned_pr_id}!" if spawned_pr_id else ""),
            tags=["jira", "engineering"]
        ))

        bucket = self._state.ticket_actors_today.setdefault(ticket_id, set())
        bucket.add(assignee)

        if self._vader:
            self._score_and_apply_sentiment(comment_text, [assignee], self._vader)

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
        backstory = self._persona_helper(reviewer, mem=self._mem, graph_dynamics=self._gd, 
                                         extra=f"You are {reviewer}, reviewing {author}'s PR: {pr_title}.")

        # Generate review comment
        agent = Agent(
            role="Code Reviewer",
            goal="Write a realistic PR review comment.",
            backstory=backstory,
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

        pr_comment = {"author": reviewer, "date": date_str,
              "timestamp": current_actor_time, "text": review_text}
        pr.setdefault("comments", []).append(pr_comment)
        
        # Write back the mutated PR (comment appended) to both stores
        pr_path = f"{self._base}/git/prs/{pr.get('pr_id', pr_id)}.json"
        import os, json as _json
        os.makedirs(os.path.dirname(pr_path), exist_ok=True)
        with open(pr_path, "w") as f:
            _json.dump(pr, f, indent=2)
        self._mem.upsert_pr(pr)   

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

        active_inc = next(
            (i for i in self._state.active_incidents
            if i.ticket_id == pr.get("linked_ticket")),
            None
        )
        if active_inc and getattr(active_inc, "causal_chain", None):
            active_inc.causal_chain.append(pr.get("pr_id", pr_id or ""))

        if self._vader:
            self._score_and_apply_sentiment(review_text, [reviewer], self._vader)

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

        ctx = self._mem.context_for_prompt(
            f"1on1 {name} {collaborator} workload", n=2, as_of_time=meeting_time_iso
        )

        def _voice_card(name: str) -> str:
            p = self._config.get("personas", {}).get(name, {})
            stress = self._gd._stress.get(name, 30)
            quirks = p.get("typing_quirks", "standard professional grammar")
            tenure = p.get("tenure", "mid")
            mood = (
                "drained, short replies" if stress > 80 else
                "a bit distracted" if stress > 60 else
                "relaxed and present"
            )
            return (
                f"{name} | Tenure: {tenure}\n"
                f"  Typing style: {quirks}\n"
                f"  Current mood: {mood}"
            )

        voice_cards = f"{_voice_card(name)}\n\n{_voice_card(collaborator)}"

        agents, tasks, prev_task = [], [], None
        for i, speaker in enumerate([name, collaborator]):
            p = self._config.get("personas", {}).get(speaker, {})
            backstory = self._persona_helper(speaker, mem=self._mem, graph_dynamics=self._gd)
            agent = Agent(
                role=f"{speaker} — {p.get('role', 'Engineer')}",
                goal=f"Have a natural 1:1 DM conversation as {speaker}.",
                backstory=backstory,
                llm=self._worker,
            )
            other = collaborator if i == 0 else name
            desc = (
                f"You are {speaker}. You are in a private Slack DM with {other}.\n\n"
                f"Both of you:\n{voice_cards}\n\n"
                f"Context: {ctx}\n\n"
                f"{'Open the conversation' if i == 0 else 'Reply naturally'}. "
                f"Topics might include workload, sprint decisions, or something "
                f"personal-professional. Use your typing quirks. 1-2 sentences. "
                f"Format: {speaker}: [message]"
            )
            task = Task(
                description=desc,
                expected_output=f"One message from {speaker} in format: {speaker}: [message]",
                agent=agent,
                context=[prev_task] if prev_task else [],
            )
            agents.append(agent)
            tasks.append(task)
            prev_task = task

        # Add 1-2 follow-up turns so the DM feels like a real back-and-forth
        for round_i in range(random.randint(1, 2)):
            for i, speaker in enumerate([name, collaborator]):
                p = self._config.get("personas", {}).get(speaker, {})
                backstory = self._persona_helper(speaker, mem=self._mem, graph_dynamics=self._gd)
                agent = Agent(
                    role=f"{speaker} — {p.get('role', 'Engineer')}",
                    goal=f"Continue the DM naturally as {speaker}.",
                    backstory=backstory,
                    llm=self._worker,
                )
                task = Task(
                    description=(
                        f"You are {speaker}. Continue the DM conversation. "
                        f"React to what was just said. Stay in character. "
                        f"Format: {speaker}: [message]"
                    ),
                    expected_output=f"One message from {speaker} in format: {speaker}: [message]",
                    agent=agent,
                    context=[prev_task] if prev_task else [],
                )
                agents.append(agent)
                tasks.append(task)
                prev_task = task

        result = str(
            Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False).kickoff()
        )

        messages = self._parse_slack_messages(
            result, [name, collaborator]
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
        slack_path, thread_id = self._save_slack(messages, channel, interaction_type="1on1")

        self._mem.log_event(SimEvent(
            type="1on1",
            timestamp=meeting_time_iso,
            day=self._state.day,
            date=date_str,
            actors=[name, collaborator],
            artifact_ids={"slack_path": slack_path, "slack_thread": thread_id},
            facts={"participants": [name, collaborator], "message_count": len(messages)},
            summary=f"1:1 between {name} and {collaborator}.",
            tags=["1on1", "slack"],
        ))

        if self._vader and messages:
            full_text = " ".join(m["text"] for m in messages)
            self._score_and_apply_sentiment(full_text, [name, collaborator], self._vader)

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
        Each participant speaks from their own persona via a dedicated Agent.
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

        chat_duration_mins  = random.randint(5, 45)
        chat_duration_hours = chat_duration_mins / 60.0
        provisional_start, _ = self._clock.sync_and_advance(
            initial_participants,
            hours=chat_duration_hours
        )
        meeting_time_iso = provisional_start.isoformat()

        seed = [collaborator] if collaborator else []
        all_actors = self._expertise_matched_participants(
            topic=ticket_title,
            seed_participants=[asker] + seed,
            as_of_time=meeting_time_iso,
            max_extras=1,
        )

        meeting_start, _ = self._clock.sync_and_advance(all_actors, hours=0)
        meeting_time_iso  = meeting_start.isoformat()

        ctx = self._mem.context_for_prompt(ticket_title, n=2, as_of_time=meeting_time_iso)

        relevant_experts = self._mem.find_confluence_experts(
            topic=ticket_title,
            score_threshold=0.75,
            n=3,
            as_of_time=meeting_time_iso,
        )
        doc_hint = (
            f"Note: the following internal documentation exists and may be "
            f"referenced naturally in this conversation:\n"
            + "\n".join(
                f"  - '{e['title']}' (written by {e['author']}, day {e['day']})"
                for e in relevant_experts
            )
            if relevant_experts else ""
        )

        # ── Build per-person voice cards ─────────────────────────────────────────
        personas = self._config.get("personas", {})

        def _voice_card(name: str) -> str:
            p      = personas.get(name, {})
            stress = self._gd._stress.get(name, 30)
            quirks = p.get("typing_quirks", "standard professional grammar")
            expertise = p.get("expertise", [])
            tenure    = p.get("tenure", "mid")

            if stress > 80:
                mood = "visibly stressed, terse replies, wants to resolve this fast"
            elif stress > 60:
                mood = "somewhat distracted but trying to help"
            else:
                mood = "engaged and happy to dig in"

            expertise_str = ", ".join(str(e) for e in expertise[:3]) if expertise else "general engineering"
            return (
                f"{name} | Tenure: {tenure} | Dept: {dept_of_name(name, self._org_chart)}\n"
                f"  Typing style: {quirks}\n"
                f"  Expertise: {expertise_str}\n"
                f"  Current mood: {mood}"
            )

        voice_cards = "\n\n".join(_voice_card(n) for n in all_actors)

        # ── One Agent per participant — sequential crew ───────────────────────────
        agents    = []
        tasks     = []
        prev_task = None

        for i, name in enumerate(all_actors):
            p         = personas.get(name, {})
            backstory = self._persona_helper(name, mem=self._mem, graph_dynamics=self._gd)

            agent = Agent(
                role=f"{name} — {p.get('tenure', 'Engineer')}",
                goal=f"Respond authentically as {name} in a Slack Q&A thread.",
                backstory=backstory,
                llm=self._worker,
            )

            if i == 0:
                # Asker opens the thread
                desc = (
                    f"You are {name}. You have a question related to your work on: {ticket_title}\n\n"
                    f"Participants in this thread:\n{voice_cards}\n\n"
                    f"Relevant context: {ctx}\n"
                    f"{doc_hint}\n\n"
                    f"Post your opening question exactly as {name} would — use your typing quirks, "
                    f"reflect your current mood, and be specific about what you're stuck on. "
                    f"1-3 sentences. Format: {name}: [message]"
                )
            else:
                # Responders reply to what came before
                desc = (
                    f"You are {name}. A colleague just asked a question in Slack about: {ticket_title}\n\n"
                    f"Participants:\n{voice_cards}\n\n"
                    f"Relevant context: {ctx}\n"
                    f"{doc_hint}\n\n"
                    f"Reply naturally as {name} would — use your typing quirks, reflect your mood. "
                    f"You may have the answer, ask a clarifying question, say you don't know but "
                    f"suggest someone else, or reference internal docs if relevant. "
                    f"1-3 sentences. Format: {name}: [message]"
                )

            task = Task(
                description=desc,
                expected_output=f"One Slack message from {name} in format: {name}: [message]",
                agent=agent,
                context=[prev_task] if prev_task else [],
            )

            agents.append(agent)
            tasks.append(task)
            prev_task = task

        result = str(
            Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False).kickoff()
        )

        messages = self._parse_slack_messages(result, all_actors)
        if not messages:
            return all_actors

        current_msg_time = datetime.fromisoformat(meeting_time_iso)
        for msg in messages:
            msg["ts"] = current_msg_time.isoformat()
            current_msg_time += timedelta(minutes=random.randint(1, 4))

        slack_path, thread_id = self._save_slack(messages, channel, interaction_type="async_question")

        active_inc = None
        if ticket_id:
            active_inc = next(
                (i for i in self._state.active_incidents if i.ticket_id == ticket_id), None
            )
            if active_inc and getattr(active_inc, "causal_chain", None):
                active_inc.causal_chain.append(thread_id)

        facts = {
            "asker":         asker,
            "channel":       channel,
            "topic":         ticket_title,
            "responders":    [a for a in all_actors if a != asker],
            "message_count": len(messages),
        }
        if active_inc and getattr(active_inc, "causal_chain", None):
            facts["causal_chain"] = active_inc.causal_chain.snapshot()

        self._mem.log_event(SimEvent(
            type="async_question",
            timestamp=meeting_time_iso,
            day=self._state.day,
            date=date_str,
            actors=all_actors,
            artifact_ids={"slack": slack_path, "slack_thread": thread_id, "jira": ticket_id or ""},
            facts=facts,
            summary=f"{asker} asked a question in #{channel} about {ticket_title[:50]}.",
            tags=["async_question", "slack"],
        ))

        if self._vader and messages:
            full_text = " ".join(m["text"] for m in messages)
            self._score_and_apply_sentiment(full_text, all_actors, self._vader)

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

        chat_duration_mins  = random.randint(5, 45)
        chat_duration_hours = chat_duration_mins / 60.0
        provisional_start, _ = self._clock.sync_and_advance(
            participants,
            hours=chat_duration_hours
        )
        meeting_time_iso = provisional_start.isoformat()

        # Augment with expertise-matched participants.  The helper injects:
        #   - the author of any published-today Confluence page on this topic
        #   - up to 1 extra person with overlapping expertise and graph proximity.
        # This replaces the previous ad-hoc topic keyword scan.
        participants = self._expertise_matched_participants(
            topic=item.description,
            seed_participants=participants,
            as_of_time=meeting_time_iso,
            max_extras=1,
        )

        meeting_start, meeting_end = self._clock.sync_and_advance(
            participants,
            hours=0,
        )
        meeting_time_iso = meeting_start.isoformat()

        ctx = self._mem.context_for_prompt(item.description, n=3, as_of_time=meeting_time_iso)

        def _voice_card(name: str) -> str:
            p = self._config.get("personas", {}).get(name, {})
            stress = self._gd._stress.get(name, 30)
            quirks = p.get("typing_quirks", "standard professional grammar")
            expertise = ", ".join(p.get("expertise", [])[:3])
            social_role = p.get("social_role", "Contributor")
            mood = (
                "terse, wants to decide fast and move on" if stress > 80 else
                "engaged but watching the clock" if stress > 60 else
                "thinking carefully, happy to explore trade-offs"
            )
            return (
                f"{name} | Role: {social_role} | Expertise: {expertise}\n"
                f"  Typing style: {quirks}\n"
                f"  Current mood: {mood}"
            )

        voice_cards = "\n\n".join(_voice_card(p) for p in participants)

        agents, tasks, prev_task = [], [], None
        # Initiator opens, then round-robin through participants for 5-8 turns total
        turn_speakers = [initiator] + (
            [participants[i % len(participants)] for i in range(1, random.randint(5, 8))]
        )

        for i, speaker in enumerate(turn_speakers):
            p = self._config.get("personas", {}).get(speaker, {})
            backstory = self._persona_helper(speaker, mem=self._mem, graph_dynamics=self._gd)
            agent = Agent(
                role=f"{speaker} — {p.get('social_role', 'Engineer')}",
                goal=f"Contribute authentically to a technical design discussion as {speaker}.",
                backstory=backstory,
                llm=self._planner,
            )
            if i == 0:
                desc = (
                    f"You are {speaker}, opening a Slack design discussion.\n\n"
                    f"Topic: {item.description}\n\n"
                    f"Participants:\n{voice_cards}\n\n"
                    f"Context: {ctx}\n\n"
                    f"Frame the problem or decision that needs to be made. "
                    f"Be specific — name constraints, risks, or the trade-off you're wrestling with. "
                    f"Use your typing quirks. 2-3 sentences max. Format: {speaker}: [message]"
                )
            else:
                others = [p for p in participants if p != speaker]
                desc = (
                    f"You are {speaker}. Respond to the design thread about: {item.description}.\n\n"
                    f"Participants:\n{voice_cards}\n\n"
                    f"React as an engineer actually working through this — raise a trade-off, "
                    f"ask 'what about X', push back, or propose a concrete next step. "
                    f"Don't just agree. Stay in character. "
                    f"Format: {speaker}: [message]"
                )
            task = Task(
                description=desc,
                expected_output=f"One message from {speaker} in format: {speaker}: [message]",
                agent=agent,
                context=[prev_task] if prev_task else [],
            )
            agents.append(agent)
            tasks.append(task)
            prev_task = task

        result = str(
            Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False).kickoff()
        )

        messages = self._parse_slack_messages(result, participants)

        depts = {dept_of_name(p, self._org_chart) for p in participants}
        if len(depts) > 1:
            dept_channel = "digital-hq"
        else:
            dept_channel = dept_of_name(initiator, self._org_chart).lower().replace(" ", "-")

        if messages:
            current_msg_time = datetime.fromisoformat(meeting_time_iso)

            for msg in messages:
                # Stamp the message with our exact, mathematically safe time
                msg["ts"] = current_msg_time.isoformat()
                
                # Add 1 to 4 random minutes so the reply looks like a human typing
                current_msg_time += timedelta(minutes=random.randint(1, 8))

        slack_path, thread_id = self._save_slack(messages, dept_channel, interaction_type="design")

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
            artifact_ids={"slack_path": slack_path, "slack_thread": thread_id, "confluence": conf_id or ""},
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

        def _voice_card(name: str) -> str:
            p = self._config.get("personas", {}).get(name, {})
            stress = self._gd._stress.get(name, 30)
            quirks = p.get("typing_quirks", "standard professional grammar")
            tenure = p.get("tenure", "mid")
            expertise = ", ".join(p.get("expertise", [])[:3])
            mood = (
                "drained, keeping answers short" if stress > 80 else
                "patient but distracted" if stress > 60 else
                "engaged and generous with their time"
            )
            return (
                f"{name} | Tenure: {tenure} | Expertise: {expertise}\n"
                f"  Typing style: {quirks}\n"
                f"  Current mood: {mood}"
            )

        voice_cards = (
            f"MENTOR:\n{_voice_card(mentor)}\n\n"
            f"MENTEE:\n{_voice_card(mentee)}"
        )

        agents, tasks, prev_task = [], [], None
        n_turns = self._turn_count([mentor, mentee], (3, 6))
        speakers = [mentor, mentee, mentor, mentee, mentor, mentee]
        for i, speaker in enumerate(speakers[:n_turns]):
            p = self._config.get("personas", {}).get(speaker, {})
            backstory = self._persona_helper(speaker, mem=self._mem, graph_dynamics=self._gd)
            is_mentor = speaker == mentor
            agent = Agent(
                role=f"{speaker} — {'Mentor' if is_mentor else 'Mentee'}",
                goal=(
                    f"Guide {mentee} thoughtfully as an experienced engineer."
                    if is_mentor else
                    f"Ask genuine questions and absorb guidance as someone still learning."
                ),
                backstory=backstory,
                llm=self._worker,
            )
            if i == 0:
                desc = (
                    f"You are {mentor}, opening a mentoring DM with {mentee}.\n\n"
                    f"{voice_cards}\n\n"
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
            task = Task(
                description=desc,
                expected_output=f"One message from {speaker} in format: {speaker}: [message]",
                agent=agent,
                context=[prev_task] if prev_task else [],
            )
            agents.append(agent)
            tasks.append(task)
            prev_task = task

        result = str(
            Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False).kickoff()
        )

        messages = self._parse_slack_messages(
            result, [mentor, mentee]
        )
        if not messages:
            return [mentor, mentee]
        
        current_msg_time = datetime.fromisoformat(meeting_time_iso)

        for msg in messages:
            # Overwrites the dummy timestamp from the parser
            msg["ts"] = current_msg_time.isoformat()
            current_msg_time += timedelta(minutes=random.randint(1, 8))

        p1, p2  = sorted([mentor, mentee])
        channel = f"dm_{p1.lower()}_{p2.lower()}"
        slack_path, thread_id = self._save_slack(messages, channel, interaction_type="mentoring")

        # Mentoring is a strong relationship signal
        self._gd.record_slack_interaction([mentor, mentee])
        self._gd.record_slack_interaction([mentor, mentee])  # double boost

        self._mem.log_event(SimEvent(
            type="mentoring",
            timestamp=meeting_time_iso,
            day=self._state.day,
            date=date_str,
            actors=[mentor, mentee],
            artifact_ids={"slack_path": slack_path, "slack_thread": thread_id},
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
        slack_path, thread_id = self._save_slack(messages, channel)
        
        # Log the simulation event for the memory store
        self._mem.log_event(SimEvent(
            type="org_collision",
            timestamp=self._clock.now("system").isoformat(),
            day=self._state.day, date=date_str,
            actors=participants,
            artifact_ids={"slack_path": slack_path, "slack_thread": thread_id},
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
            result, [asker, collaborator]
        )

        if messages:
            current_msg_time = datetime.fromisoformat(timestamp)

            for msg in messages:
                # Stamp the message with our exact, mathematically safe time
                msg["ts"] = current_msg_time.isoformat()
                
                # Add 1 to 4 random minutes so the reply looks like a human typing
                current_msg_time += timedelta(minutes=random.randint(1, 4))

            slack_path, thread_id = self._save_slack(messages, channel, interaction_type="blocker")
            self._gd.record_slack_interaction([asker, collaborator])

            active_inc = next((i for i in self._state.active_incidents if i.ticket_id == ticket_id), None)
            if active_inc and getattr(active_inc, "causal_chain", None):
                active_inc.causal_chain.append(thread_id)
                
            facts = {"blocker_reason": blocker_text}
            if active_inc and getattr(active_inc, "causal_chain", None):
                facts["causal_chain"] = active_inc.causal_chain.snapshot()

            self._mem.log_event(SimEvent(
                type="blocker_flagged",
                day=self._state.day,
                date=date_str,
                timestamp=timestamp,
                actors=[asker, collaborator],
                artifact_ids={"slack_path": slack_path, "slack_thread": thread_id},
                facts=facts,
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
            timestamp=timestamp
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

        edge_weight = edges.get(colleagues[0], {}).get("weight", 0.5) if colleagues else 0.5
        stress_avg = sum(self._gd._stress.get(n, 30) for n in participants) / len(participants)
        hour = thread_start.hour

        interests_str = ", ".join(set(participant_interests)) if participant_interests else "general life topics"

        topic_agent = Agent(
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
        topic = str(Crew(agents=[topic_agent], tasks=[topic_task], verbose=False).kickoff()).strip()

        # ── Build rich per-person voice cards from personas ───────────────────
        personas = self._config.get("personas", {})

        def _voice_card(name: str) -> str:
            p = personas.get(name, {})
            stress = self._gd._stress.get(name, 30)
            quirks = p.get("typing_quirks", "standard professional grammar")
            interests = p.get("interests", p.get("expertise", []))  # fall back to expertise if no interests
            tenure = p.get("tenure", "mid")
            social_role = p.get("social_role", "Contributor")

            if stress > 80:
                mood = "visibly drained, short replies, clearly wants this to be over quickly"
            elif stress > 60:
                mood = "a bit distracted, somewhat engaged but mind is elsewhere"
            else:
                mood = "relaxed and happy to take a break"

            interests_str = ", ".join(str(i) for i in interests[:3]) if interests else "general topics"
            return (
                f"{name} | Tenure: {tenure} | Role: {social_role}\n"
                f"  Typing style: {quirks}\n"
                f"  Personal interests: {interests_str}\n"
                f"  Current mood: {mood}"
            )

        voice_cards = "\n\n".join(_voice_card(n) for n in participants)

        # ── One agent per participant — each speaks in their own voice ────────
        agents = []
        tasks = []
        prev_task = None

        for i, name in enumerate(participants):
            p = personas.get(name, {})
            backstory = self._persona_helper(name, mem=self._mem, graph_dynamics=self._gd)

            agent = Agent(
                role=f"{name} — {p.get('social_role', 'Team Member')}",
                goal=f"Chat naturally as {name} would in a casual Slack conversation.",
                backstory=backstory,
                llm=self._worker,
            )

            if i == 0:
                # Initiator — starts the thread
                desc = (
                    f"You are {name}. You just opened a Slack message to your colleagues "
                    f"about: {topic}.\n\n"
                    f"Participants in this chat:\n{voice_cards}\n\n"
                    f"Write your opening message exactly as {name} would — use your typing "
                    f"quirks, reflect your mood, and make it feel spontaneous. "
                    f"1-2 sentences max. Format: {name}: [message]"
                )
            else:
                # Responders — react to what came before
                desc = (
                    f"You are {name}. You just received a Slack message from your colleague "
                    f"about: {topic}.\n\n"
                    f"Participants:\n{voice_cards}\n\n"
                    f"Reply naturally as {name} would. Use your typing quirks and reflect "
                    f"your current mood. Keep it casual and non-work. "
                    f"1-2 sentences. Format: {name}: [message]"
                )

            task = Task(
                description=desc,
                expected_output=f"One Slack message from {name} in format: {name}: [message]",
                agent=agent,
                context=[prev_task] if prev_task else [],
            )

            agents.append(agent)
            tasks.append(task)
            prev_task = task

        result = str(
            Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False).kickoff()
        )

        messages = self._parse_slack_messages(result, participants)

        if messages:
            channel = (
                "random" if len(participants) > 2
                else f"dm_{sorted(participants)[0].lower()}_{sorted(participants)[1].lower()}"
            )

            slack_path, thread_id = self._mem.log_slack_messages(
                channel=channel,
                messages=messages,
                export_dir=Path(self._base),
            )

            self._gd.record_slack_interaction(participants)

            self._mem.log_event(SimEvent(
                type="watercooler_chat",
                timestamp=thread_start_iso,
                day=self._state.day,
                date=date_str,
                actors=participants,
                artifact_ids={"slack_thread": thread_id, "slack_path":   slack_path},
                facts={"topic": topic, "message_count": len(messages)},
                summary=f"{target_actor} got distracted chatting about {topic} with {len(participants)-1} others.",
                tags=["watercooler", "slack", "distraction"],
            ))

            logger.info(f"    [dim]☕ Distraction: {target_actor} pulled into chat about {topic}[/dim]")

    # ─── LOW-LEVEL UTILITIES ──────────────────────────────────────────────────

    def _parse_slack_messages(
        self,
        raw:         str,
        valid_names: List[str],
        cadence:     str = "normal", # Use the cadence ranges from sim_clock
    ) -> List[dict]:
        messages = []
        
        # 1. Sync all participants to the exact same starting point
        # This guarantees the conversation starts AFTER whatever they were just doing
        current_time = self._clock.sync_and_tick(valid_names, min_mins=0, max_mins=2)

        for line in raw.split("\n"):
            if ":" not in line:
                continue
            name, text = line.split(":", 1)
            name = name.strip()
            
            if name in valid_names:
                # 2. Advance the clock forward safely using sim_clock
                # This ensures the next message is always chronologically after this one
                current_time = self._clock.tick_message(valid_names, cadence=cadence)
                
                messages.append({
                    "user": name,
                    "text": text.strip(),
                    "ts":   current_time.isoformat(),
                })
        return messages

    def _save_slack(self, messages: List[dict], channel: str, interaction_type: str = "general") -> Tuple[str, str]:
        """Write Slack messages to disk + MongoDB. Returns export path."""
        date_str = str(self._state.current_date.date())
        # Ensure every message has the date field log_slack_messages needs
        for m in messages:
            m.setdefault("date", date_str)
            
        slack_path, thread_id = self._mem.log_slack_messages(    # handles disk + MongoDB + path
            channel=channel,
            messages=messages,
            export_dir=Path(self._base)
        )

        if messages:
            # Concatenate the full conversation so the RAG context preserves the flow
            full_transcript = "\n".join(f"{m['user']}: {m['text']}" for m in messages)
            start_timestamp = messages[0].get("ts", self._clock.now("system").isoformat())
            
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
                    "message_count": len(messages)
                }
            )

        return slack_path, thread_id

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

        self._mem.upsert_ticket(ticket)

    def _emit_bot_message(
        self, channel: str, bot_name: str, text: str, timestamp: str
    ) -> None:
        """Unified 4-arg signature matching flow.py._emit_bot_message."""
        date_str = str(self._state.current_date.date())
        msg = {
            "user":   bot_name,
            "email":  f"{bot_name.lower()}@bot.{self._domain}",
            "text":   text,
            "ts":     timestamp,
            "date":   date_str,
            "is_bot": True,
        }
        self._mem.log_slack_messages(
            channel=channel,
            messages=[msg],
            export_dir=Path(self._base),
        )

    def _find_ticket(self, ticket_id: Optional[str]) -> Optional[dict]:
        if not ticket_id: return None
        return self._mem.get_ticket(ticket_id)

    def _find_pr(self, pr_id: Optional[str]) -> Optional[dict]:
        if not pr_id:
            return None
        return self._mem._prs.find_one({"pr_id": pr_id})

    def _find_reviewable_pr(self, reviewer: str) -> Optional[dict]:
        """Find an open PR where this person is listed as a reviewer."""
        prs = self._mem.get_reviewable_prs_for(reviewer)
        return random.choice(prs) if prs else None

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
        avg_stress = sum(
            self._gd._stress.get(n, 30) for n in participants
        ) / len(participants)

        if avg_stress > 80:
            return default_range[0]                          # floor — terse, get-it-done
        elif avg_stress > 60:
            return random.randint(*default_range[:2])        # low end of range
        else:
            return random.randint(*default_range)            # full range

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — dept lookup without importing flow globals
# ─────────────────────────────────────────────────────────────────────────────

def dept_of_name(name: str, org_chart: Dict[str, List[str]]) -> str:
    for dept, members in org_chart.items():
        if name in members:
            return dept
    return "Unknown"