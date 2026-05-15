"""Exercise 4 - Structured SQLite audit trail + durable checkpointer."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.db import db_conn, db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import ainvoke_structured
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"
INTERRUPT_ACTIONS = {"human_approval", "escalate"}


async def audit(state: ReviewState, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the audit_events table.

    Pending HITL rows are emitted before interrupt(). LangGraph re-runs the node
    when it resumes, so those pending rows are de-duplicated by thread/action.
    """
    if entry.action in INTERRUPT_ACTIONS and entry.decision == "pending":
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT id
                  FROM audit_events
                 WHERE thread_id = ?
                   AND pr_url = ?
                   AND action = ?
                   AND decision = 'pending'
                   AND reason = ?
                 LIMIT 1
                """,
                (
                    state["thread_id"],
                    state["pr_url"],
                    entry.action,
                    entry.reason,
                ),
            ) as cur:
                if await cur.fetchone() is not None:
                    return

    await write_audit_event(
        thread_id=state["thread_id"],
        pr_url=state["pr_url"],
        entry=entry,
    )


def _elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _reviewer_id() -> str | None:
    return (
        os.environ.get("GITHUB_USER")
        or os.environ.get("GITHUB_ACTOR")
        or os.environ.get("REVIEWER_ID")
        or os.environ.get("USERNAME")
        or os.environ.get("USER")
    )


def _analysis_prompt() -> str:
    return (
        "You are a senior code reviewer. Return structured output only. "
        "Review the unified diff for correctness, security, test coverage, data "
        "migrations, and maintainability. Calibrate confidence carefully: high "
        "confidence is only for tiny safe changes; medium confidence is for "
        "ordinary changes that need human approval; low confidence is for "
        "security-sensitive, ambiguous, or high-blast-radius work. If confidence "
        f"is below {ESCALATE_THRESHOLD:.0%}, populate escalation_questions with "
        "2-4 specific questions that reference the relevant file, code area, or "
        "risk. Auth, password hashing, token storage, SQL construction, cloud "
        "sync, secrets, or hard-coded identities should usually be low confidence "
        "unless the diff clearly proves safeguards and tests."
    )


def _route_decision_label(decision: str) -> str:
    if decision == "auto_approve":
        return "auto"
    if decision == "escalate":
        return "escalate"
    return "pending"


async def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]-> fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]OK[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {
        "pr_title": pr.title,
        "pr_author": pr.author,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


async def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]-> analyze[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a = await ainvoke_structured(PRAnalysis, [
            {"role": "system", "content": _analysis_prompt()},
            {
                "role": "user",
                "content": (
                    f"Title: {state['pr_title']}\n"
                    f"Author: {state.get('pr_author', 'unknown')}\n"
                    f"Files: {', '.join(state.get('pr_files', []))}\n\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    console.print(f"  [green]OK[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=a.confidence_reasoning,
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"analysis": a}


async def node_route(state: ReviewState) -> dict:
    console.print("[cyan]-> route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]OK[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision=_route_decision_label(decision),
        reason=f"Routed to {decision}",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"decision": decision}


async def node_human_approval(state: ReviewState) -> dict:
    console.print("[cyan]-> human_approval[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]
    pending_reason = (
        "Waiting for final approval of refined escalation review"
        if state.get("escalation_answers")
        else "Waiting for reviewer approval of initial analysis"
    )
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id(),
        decision="pending",
        reason=pending_reason,
        execution_time_ms=_elapsed_ms(t0),
    ))

    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:4000],
    })
    if not isinstance(resp, dict):
        resp = {"choice": "reject", "feedback": "Invalid resume payload"}

    choice = str(resp.get("choice", "reject")).lower()
    if choice not in {"approve", "reject", "edit"}:
        choice = "reject"
    feedback = str(resp.get("feedback") or "")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id(),
        decision=choice,
        reason=feedback or f"Reviewer chose {choice}",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"human_choice": choice, "human_feedback": feedback}


def _render_comment_body(state: ReviewState) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    if a.risk_factors:
        lines.append("**Risk factors**")
        for risk in a.risk_factors:
            lines.append(f"- {risk}")
        lines.append("")
    if a.comments:
        lines.append("**Review comments**")
        for c in a.comments:
            lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` - {c.body}")
    else:
        lines.append("No blocking comments found.")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state: ReviewState) -> tuple[str, str, str | None]:
    body = _render_comment_body(state)
    try:
        url = post_review_comment(state["pr_url"], body)
        console.print(f"  [green]OK[/green] posted comment to {state['pr_url']}")
        return "committed", body, url
    except Exception as e:
        console.print(f"  [red]FAILED[/red] post failed: {e}")
        return "commit_failed", body, None


async def _auto_edit_analysis(state: ReviewState) -> PRAnalysis:
    feedback = state.get("human_feedback") or "Make the review clearer and more actionable."
    with console.status("[dim]LLM rewriting review from reviewer edit...[/dim]"):
        edited = await ainvoke_structured(PRAnalysis, [
            {
                "role": "system",
                "content": (
                    "Rewrite the automated PR review using the human edit request. "
                    "Preserve valid findings, remove anything the human rejected, "
                    "make the final comment concise and actionable, and return "
                    "structured output."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Human edit request:\n{feedback}\n\n"
                    f"Current analysis:\n{state['analysis'].model_dump_json(indent=2)}\n\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    return edited


async def node_commit(state: ReviewState) -> dict:
    console.print("[cyan]-> commit[/cyan]")
    t0 = time.monotonic()
    working_state: ReviewState = dict(state)  # type: ignore[assignment]
    choice = state.get("human_choice")
    action = "rejected"
    body: str | None = None
    url: str | None = None
    decision = "reject"
    reason = "No comment posted"

    if choice == "reject":
        console.print(f"  [yellow]SKIP[/yellow] skipping comment (choice={choice})")
    else:
        if choice == "edit":
            edited = await _auto_edit_analysis(state)
            working_state["analysis"] = edited
            await audit(state, AuditEntry(
                agent_id=AGENT_ID,
                action="auto_edit",
                confidence=edited.confidence,
                risk_level=risk_level_for(edited.confidence),
                reviewer_id=_reviewer_id(),
                decision="edit",
                reason=state.get("human_feedback") or "Auto-edited review from human feedback",
                execution_time_ms=_elapsed_ms(t0),
            ))
            decision = "edit"
        elif choice == "approve":
            decision = "approve"
        elif state.get("escalation_answers"):
            decision = "escalate"
        else:
            decision = "auto"

        action, body, url = _post(working_state)
        if action == "committed" and state.get("escalation_answers") and choice is None:
            action = "committed_after_escalation"
        reason = "Posted PR review comment" if action != "commit_failed" else "Posting PR review comment failed"

    a = working_state["analysis"]
    await audit(working_state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id() if choice else None,
        decision=decision,
        reason=reason,
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {
        "analysis": a,
        "posted_comment_body": body,
        "posted_comment_url": url,
        "final_action": action,
    }


async def node_auto_approve(state: ReviewState) -> dict:
    console.print("[cyan]-> auto_approve[/cyan] [dim]high confidence - posting directly[/dim]")
    t0 = time.monotonic()
    action, body, url = _post(state)
    a = state["analysis"]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="auto",
        reason="High confidence review posted without human approval",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {
        "posted_comment_body": body,
        "posted_comment_url": url,
        "final_action": f"auto_{action}",
    }


async def node_escalate(state: ReviewState) -> dict:
    console.print("[cyan]-> escalate[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]
    questions = a.escalation_questions or [
        "What is the intended production behavior of this PR?",
        "Are there migration, security, or test constraints not visible in the diff?",
    ]

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id(),
        decision="pending",
        reason="Waiting for reviewer answers to escalation questions",
        execution_time_ms=_elapsed_ms(t0),
    ))

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
        "diff_preview": state["pr_diff"][:4000],
    })
    if not isinstance(answers, dict):
        answers = {}

    answered = sum(1 for value in answers.values() if str(value).strip())
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id(),
        decision="escalate",
        reason=f"Reviewer answered {answered}/{len(questions)} escalation questions",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"escalation_answers": answers}


async def node_synthesize(state: ReviewState) -> dict:
    console.print("[cyan]-> synthesize[/cyan]")
    t0 = time.monotonic()
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    original = state["analysis"]
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined = await ainvoke_structured(PRAnalysis, [
            {
                "role": "system",
                "content": (
                    "Refine the PR review with the reviewer's answers. Keep confirmed "
                    "issues, remove concerns that the answers resolve, add any new "
                    "risks implied by the answers, and update confidence honestly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original analysis:\n{original.model_dump_json(indent=2)}\n\n"
                    f"Reviewer Q&A:\n{qa}\n\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    console.print(f"  [green]OK[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        reviewer_id=_reviewer_id(),
        decision="pending" if state.get("require_final_approval") else "escalate",
        reason=refined.confidence_reasoning,
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"analysis": refined}


def _after_synthesize(state: ReviewState) -> str:
    return "human_approval" if state.get("require_final_approval") else "commit"


def build_graph(checkpointer):
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval", node_human_approval),
        ("commit", node_commit),
        ("escalate", node_escalate),
        ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    g.add_edge("escalate", "synthesize")
    g.add_conditional_edges(
        "synthesize",
        _after_synthesize,
        {"human_approval": "human_approval", "commit": "commit"},
    )
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload: dict[str, Any]):
    kind = payload["kind"]
    if kind == "approval_request":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Approval conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("approve/reject/edit? ").strip().lower()
        feedback = console.input("Feedback: ").strip() if choice != "approve" else ""
        return {"choice": choice, "feedback": feedback}
    if kind == "escalation":
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        if payload.get("risk_factors"):
            console.print("[bold]Risk factors[/bold]")
            for risk in payload["risk_factors"]:
                console.print(f"- {risk}")
        return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}
    raise ValueError(kind)


async def run(pr_url: str, thread_id: str | None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 - SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            result = await app.ainvoke(Command(resume=handle_interrupt(payload)), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        if result.get("posted_comment_url"):
            console.print(f"comment_url = {result['posted_comment_url']}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    parser.add_argument("--thread", help="Resume an existing thread")
    args = parser.parse_args()
    asyncio.run(run(args.pr, args.thread))


if __name__ == "__main__":
    main()
